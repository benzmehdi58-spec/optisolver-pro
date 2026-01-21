import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import copy
import math
import uuid
import random

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="OptiSolver Pro",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #4B4B4B; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; color: #0068C9; margin-top: 2rem; border-bottom: 2px solid #0068C9; padding-bottom: 0.5rem; }
    .stButton>button { width: 100%; background-color: #0068C9; color: white; height: 3rem; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER: HIERARCHICAL TREE LAYOUT
# ==========================================
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Custom layout function to position nodes in a perfect tree structure.
    No external Graphviz dependency required.
    """
    if not nx.is_tree(G):
        return nx.spring_layout(G)

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G))) 
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children) != 0:
            dx = width / len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                                     vert_loc=vert_loc-vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

# ==========================================
# 3. CUSTOM SOLVER (FROM SCRATCH)
# ==========================================

class LinearOptimizationSolver:
    def __init__(self):
        pass

    # --- CORE SIMPLEX METHODS ---
    def _pivot_step(self, tableau, pivot_row_idx, pivot_col_idx):
        pivot_val = tableau[pivot_row_idx][pivot_col_idx]
        tableau[pivot_row_idx] = [x / pivot_val for x in tableau[pivot_row_idx]]
        for i in range(len(tableau)):
            if i != pivot_row_idx:
                factor = tableau[i][pivot_col_idx]
                tableau[i] = [curr - factor * piv for curr, piv in zip(tableau[i], tableau[pivot_row_idx])]

    def _restore_feasibility(self, tableau):
        while True:
            rhs_col = [row[-1] for row in tableau[:-1]]
            min_b = min(rhs_col)
            if min_b >= -1e-9:
                return True 
            
            pivot_row_idx = rhs_col.index(min_b)
            row_vals = tableau[pivot_row_idx][:-1]
            if all(val >= -1e-9 for val in row_vals): return False 
            
            pivot_col_idx = -1
            min_ratio = float('inf')
            objective_row = tableau[-1][:-1]
            for j, val in enumerate(row_vals):
                if val < -1e-9:
                    ratio = abs(objective_row[j] / val)
                    if ratio < min_ratio:
                        min_ratio = ratio
                        pivot_col_idx = j
            if pivot_col_idx == -1: return False
            self._pivot_step(tableau, pivot_row_idx, pivot_col_idx)

    def _solve_simplex_core(self, tableau):
        if not self._restore_feasibility(tableau): return None 
        num_rows = len(tableau)
        while True:
            last_row = tableau[-1][:-1]
            max_val = max(last_row)
            if max_val <= 1e-9: break
            pivot_col_idx = last_row.index(max_val)
            min_ratio, pivot_row_idx = float('inf'), -1
            for i in range(num_rows - 1): 
                rhs = tableau[i][-1]
                col_val = tableau[i][pivot_col_idx]
                if col_val > 1e-9:
                    ratio = rhs / col_val
                    if ratio < min_ratio:
                        min_ratio = ratio
                        pivot_row_idx = i
            if pivot_row_idx == -1: return None 
            self._pivot_step(tableau, pivot_row_idx, pivot_col_idx)
        return tableau

    def _extract_solution(self, tableau, num_vars):
        num_rows = len(tableau)
        solution = []
        for col_idx in range(num_vars):
            col_values = [tableau[r][col_idx] for r in range(num_rows)]
            ones, zeros, row_of_one = 0, 0, -1
            for r_idx, val in enumerate(col_values):
                if abs(val - 1.0) < 1e-5:
                    ones += 1
                    row_of_one = r_idx
                elif abs(val) < 1e-5:
                    zeros += 1
            if ones == 1 and (zeros == num_rows - 1):
                solution.append(tableau[row_of_one][-1])
            else:
                solution.append(0.0)
        return solution

    # --- PUBLIC API ---
    def solve_lp(self, c, A, b, mode='max'):
        c_work = c[:]
        if mode == 'min': c_work = [-1 * x for x in c_work]
        num_vars = len(c_work)
        num_constraints = len(b)
        tableau = []
        for i in range(num_constraints):
            row = A[i][:] + [0] * num_constraints + [b[i]]
            row[num_vars + i] = 1 
            tableau.append(row)
        tableau.append(c_work[:] + [0] * num_constraints + [0])
        final_tableau = self._solve_simplex_core(tableau)
        if final_tableau is None: return -float('inf'), []
        opt_val = -final_tableau[-1][-1]
        if mode == 'min': opt_val = -opt_val
        vars_val = self._extract_solution(final_tableau, num_vars)
        return opt_val, vars_val

    def solve_ilp_branch_and_bound(self, c, A, b, mode='max', binary_only=False):
        tree_graph = nx.DiGraph()
        best_integer_obj = -float('inf')
        best_solution = None
        
        current_A = copy.deepcopy(A)
        current_b = copy.deepcopy(b)
        
        if binary_only:
            num_vars = len(c)
            for i in range(num_vars):
                row = [0] * num_vars
                row[i] = 1
                current_A.append(row)
                current_b.append(1)

        root_id = "Root"
        stack = [(current_A, current_b, None, "Start")]
        
        c_internal = c[:]
        if mode == 'min':
            c_internal = [-1 * x for x in c]

        node_counter = 0

        while stack:
            curr_A, curr_b, parent_id, desc = stack.pop()
            
            node_id = str(uuid.uuid4())[:8]
            if node_counter == 0: 
                node_id = root_id
            
            if parent_id is not None:
                tree_graph.add_edge(parent_id, node_id, label=desc)
            
            node_counter += 1

            # 1. Solve Relaxation
            num_vars = len(c_internal)
            num_cons = len(curr_b)
            tableau = []
            for i in range(num_cons):
                row = curr_A[i][:] + [0] * num_cons + [curr_b[i]]
                row[num_vars + i] = 1 
                tableau.append(row)
            tableau.append(c_internal[:] + [0] * num_cons + [0])
            
            final_tableau = self._solve_simplex_core(tableau)
            
            if final_tableau is None:
                tree_graph.add_node(node_id, label="Infeasible", color="#ffcccc", shape="s")
                continue 
            
            node_obj = -final_tableau[-1][-1]
            node_vars = self._extract_solution(final_tableau, num_vars)
            
            vars_str = ", ".join([f"{v:.2f}" for v in node_vars])
            label_str = f"Z={node_obj:.2f}\n[{vars_str}]"
            
            if node_obj <= best_integer_obj + 1e-6:
                tree_graph.add_node(node_id, label=f"Pruned\n{label_str}", color="#e0e0e0", shape="s")
                continue 

            non_int_idx = -1
            non_int_val = 0
            for i, val in enumerate(node_vars):
                if abs(val - round(val)) > 1e-5:
                    non_int_idx = i
                    non_int_val = val
                    break
            
            if non_int_idx == -1:
                if node_obj > best_integer_obj:
                    best_integer_obj = node_obj
                    best_solution = node_vars
                    tree_graph.add_node(node_id, label=f"**INT**\n{label_str}", color="#90ee90", shape="s")
                else:
                    tree_graph.add_node(node_id, label=f"Sub-optimal\n{label_str}", color="#d3ffd3", shape="s")
            else:
                tree_graph.add_node(node_id, label=f"Fractional\n{label_str}", color="#add8e6", shape="s")
                val_floor = math.floor(non_int_val)
                val_ceil = math.ceil(non_int_val)
                
                # Branch 1
                new_A1 = copy.deepcopy(curr_A)
                new_b1 = copy.deepcopy(curr_b)
                row_leq = [0] * num_vars
                row_leq[non_int_idx] = 1
                new_A1.append(row_leq)
                new_b1.append(val_floor)
                stack.append((new_A1, new_b1, node_id, f"x{non_int_idx+1}≤{val_floor}"))
                
                # Branch 2
                new_A2 = copy.deepcopy(curr_A)
                new_b2 = copy.deepcopy(curr_b)
                row_geq = [0] * num_vars
                row_geq[non_int_idx] = -1
                new_A2.append(row_geq)
                new_b2.append(-val_ceil)
                stack.append((new_A2, new_b2, node_id, f"x{non_int_idx+1}≥{val_ceil}"))

        final_obj = best_integer_obj
        if mode == 'min': final_obj = -final_obj
        return final_obj, best_solution, tree_graph

# ==========================================
# 4. VISUALIZATION FUNCTIONS
# ==========================================

def plot_2d_feasible_region(c, A, b, maximize):
    if len(c) != 2: return None
    fig, ax = plt.subplots(figsize=(8, 6))
    max_b = max(b) if b else 10
    limit = max(20, max_b * 1.5)
    x = np.linspace(0, limit, 400)
    y = np.linspace(0, limit, 400)
    X, Y = np.meshgrid(x, y)
    feasible_mask = (X >= 0) & (Y >= 0)
    for i in range(len(b)):
        a_val, b_val = A[i][0], A[i][1]
        rhs = b[i]
        feasible_mask &= (a_val * X + b_val * Y <= rhs + 1e-5)
        if b_val != 0:
            y_line = (rhs - a_val * x) / b_val
            ax.plot(x, y_line, label=f'{a_val}x₁ + {b_val}x₂ ≤ {rhs}', linewidth=2)
        else:
            ax.axvline(x=rhs/a_val, label=f'{a_val}x₁ ≤ {rhs}', linewidth=2)
    ax.imshow(feasible_mask.astype(int), extent=(x.min(), x.max(), y.min(), y.max()), 
              origin="lower", cmap="Greys", alpha=0.3)
    ax.set_xlim(0, limit); ax.set_ylim(0, limit)
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
    ax.set_title('Feasible Region'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
    return fig

def plot_bb_tree(graph):
    if graph is None or graph.number_of_nodes() == 0:
        return None
    try:
        root = [n for n, d in graph.in_degree() if d == 0][0]
    except IndexError:
        root = list(graph.nodes())[0]

    pos = hierarchy_pos(graph, root)

    plt.figure(figsize=(10, 6))
    
    colors = [data.get('color', 'lightgray') for _, data in graph.nodes(data=True)]
    labels = nx.get_node_attributes(graph, 'label')
    edge_labels = nx.get_edge_attributes(graph, 'label')
    
    nx.draw(graph, pos, with_labels=True, labels=labels, 
            node_color=colors, node_size=3000, font_size=8, 
            node_shape="s", edge_color="#555555", width=1.5, arrowsize=15)
    
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)
    return plt.gcf()

# ==========================================
# 5. UI MAIN
# ==========================================

def main():
    st.markdown("<h1 class='main-header'>📐 OptiSolver Pro</h1>", unsafe_allow_html=True)
    st.markdown("Solve LP and IP problems with **Tree Visualization**.")

    with st.sidebar:
        st.header("⚙️ Settings")
        opt_type = st.selectbox("Type", ["Maximize", "Minimize"], index=0)
        num_vars = st.number_input("Variables", 1, 10, 2)
        num_constraints = st.number_input("Constraints", 1, 10, 2)
        st.markdown("---")
        solver_method = st.radio("Solver", ["Simplex (Continuous)", "Branch and Bound (Integer)"])
        is_binary = False
        if solver_method == "Branch and Bound (Integer)":
            is_binary = st.checkbox("Binary Variables {0,1} Only", value=False)

        if st.button("🔄 Reset"): st.rerun()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Objective Function")
        c_coeffs = []
        cols_obj = st.columns(num_vars)
        for i in range(num_vars):
            val = cols_obj[i].number_input(f"C{i+1}", value=1.0, key=f"c_{i}")
            c_coeffs.append(val)

    with col2:
        st.subheader("Constraints")
        A_coeffs, b_vals = [], []
        for i in range(num_constraints):
            c_row = st.columns(num_vars + 2)
            row_coeffs = []
            for j in range(num_vars):
                val = c_row[j].number_input(f"a_{i}_{j}", value=1.0 if i==j else 0.0, key=f"a_{i}_{j}", label_visibility="collapsed")
                row_coeffs.append(val)
            c_row[num_vars].markdown("≤")
            rhs = c_row[num_vars+1].number_input("RHS", value=10.0, key=f"b_{i}", label_visibility="collapsed")
            A_coeffs.append(row_coeffs); b_vals.append(rhs)

    # --- RESTORED CONSTRAINT DISPLAY ---
    st.markdown("<div class='sub-header'>📝 Formulation</div>", unsafe_allow_html=True)
    
    # Display Objective Function
    st.latex(f"\\text{{{opt_type}}} Z = " + " + ".join([f"{c}x_{{{i+1}}}" for i, c in enumerate(c_coeffs)]))
    
    # Display Constraints (FIXED)
    latex_constraints = ""
    for i in range(num_constraints):
        lhs = " + ".join([f"{A_coeffs[i][j]}x_{{{j+1}}}" for j in range(num_vars)])
        latex_constraints += f"{lhs} \\leq {b_vals[i]} \\\\"
    st.latex(f"\\text{{Subject to:}} \\\\ {latex_constraints}")
    
    # Display Domain
    if solver_method == "Simplex (Continuous)":
        st.latex("x_i \\geq 0")
    elif is_binary:
        st.latex("x_i \\in \\{0, 1\\}")
    else:
        st.latex("x_i \\in \\mathbb{Z}^+")

    if st.button("🚀 Solve Problem"):
        st.markdown("<div class='sub-header'>📊 Results</div>", unsafe_allow_html=True)
        solver = LinearOptimizationSolver()
        mode = 'max' if opt_type == "Maximize" else 'min'

        try:
            if solver_method == "Simplex (Continuous)":
                val, x_res = solver.solve_lp(c_coeffs, A_coeffs, b_vals, mode=mode)
                c1, c2 = st.columns(2)
                with c1:
                    if val == -float('inf'): st.error("Infeasible/Unbounded")
                    else:
                        st.success("Optimal Found!")
                        st.metric("Objective", f"{val:.4f}")
                        st.table(pd.DataFrame({"Var": [f"x{i+1}" for i in range(num_vars)], "Val": x_res}))
                with c2:
                    if num_vars == 2:
                        fig = plot_2d_feasible_region(c_coeffs, A_coeffs, b_vals, mode=='max')
                        if fig: st.pyplot(fig)
            else:
                val, x_res, tree = solver.solve_ilp_branch_and_bound(c_coeffs, A_coeffs, b_vals, mode=mode, binary_only=is_binary)
                c1, c2 = st.columns([1, 2])
                with c1:
                    if val == -float('inf'): st.warning("No solution found.")
                    else:
                        st.success("Integer Solution!")
                        st.metric("Objective", f"{val:.4f}")
                        st.table(pd.DataFrame({"Var": [f"x{i+1}" for i in range(num_vars)], "Val": x_res}))
                with c2:
                    st.write("**Search Tree**")
                    if tree:
                        fig = plot_bb_tree(tree)
                        if fig: st.pyplot(fig)
        except Exception as e: st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

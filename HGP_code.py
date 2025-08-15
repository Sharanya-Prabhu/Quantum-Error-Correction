from typing import Tuple, Set, Dict, List
import stim
import numpy as np
import networkx as nx
import random

# classes of surface codes

class HGPCode:
    def __init__(self, Ha: np.ndarray, Hb: np.ndarray, d: int, shift: list[int] = [0,0]):
        self.Ha = np.array(Ha)
        self.Hb = np.array(Hb)
        self.na, self.ma = self.Ha.shape[1], self.Ha.shape[0]
        self.nb, self.mb = self.Hb.shape[1], self.Hb.shape[0]
        self.Hx, self.Hz = self._HGP_code()

        self.n_data_qubits = self.na * self.nb + self.ma * self.mb
        self.n_x_stabs = self.ma * self.nb
        self.n_z_stabs = self.na * self.mb

        self.z_ancilla_offset = self.n_data_qubits
        self.x_ancilla_offset = self.n_data_qubits + self.n_z_stabs
        self.total_qubits = self.n_data_qubits + self.n_x_stabs + self.n_z_stabs

        if d % 2 == 0: # d is an odd integer
            raise ValueError("Code distance d must be an odd integer.")
        self.d = d

        self.data_coords = set()
        self.ancilla_coords_x = set()
        self.ancilla_coords_z = set()
        self.stabilizers = []
        self.logical_X = []
        self.logical_Z = []

        self._generate_HGP_code()
    
    def _HGP_code(self):
        # Constructs the HGP code parity check matrices
        Hx = np.kron(self.Ha, np.eye(self.nb, dtype=int))
        Hx = np.concatenate((Hx, np.kron(np.eye(self.ma, dtype=int), self.Hb.T)), axis=1)

        Hz = np.kron(np.eye(self.na, dtype=int), self.Hb)
        Hz = np.concatenate((Hz, np.kron(self.Ha.T, np.eye(self.mb, dtype=int))), axis=1)

        return Hx, Hz
    
    def _generate_HGP_code(self):
        self.data_coords_1 = [(j, i) for i in range(self.na) for j in range(self.nb)]
        self.data_coords_2 = [(self.nb + j + 2, self.na + i + 2) for i in range(self.ma) for j in range(self.mb)]

        self.ancilla_coords_z = [(self.nb + j + 2, i) for i in range(self.na) for j in range(self.mb)]
        self.ancilla_coords_x = [(j, self.na + i + 2) for i in range(self.ma) for j in range(self.nb)]

        self.qubit_coords = HGPCode.sort_coords(set(self.data_coords_1) | set(self.data_coords_2) | set(self.ancilla_coords_z) | set(self.ancilla_coords_x))
        self.data_coords = HGPCode.sort_coords(set(self.data_coords_1) | set(self.data_coords_2))
        self.ancilla_coords = HGPCode.sort_coords(set(self.ancilla_coords_z) | set(self.ancilla_coords_x))
        self.data_coords_1 = HGPCode.sort_coords(self.data_coords_1)
        self.data_coords_2 = HGPCode.sort_coords(self.data_coords_2)
        self.ancilla_coords_x = HGPCode.sort_coords(self.ancilla_coords_x)
        self.ancilla_coords_z = HGPCode.sort_coords(self.ancilla_coords_z)

        ordered_coords = (self.data_coords[:self.na * self.nb] + self.data_coords[self.na * self.nb:] +
                           self.ancilla_coords_z + self.ancilla_coords_x)
        
        self.coord_to_index = {coord: i for i, coord in enumerate(ordered_coords)}
        self.index_to_coord = {i: coord for coord, i in self.coord_to_index.items()}

        for i in range(self.na):
            for j in range(self.mb):
                anc = (self.nb + j + 2, i)
                data_indices = []
                for k in range(self.nb):
                    if self.Hb[j, k] == 1:
                        data_indices.append((k, i))
                for l in range(self.ma):
                    if self.Ha[l, i] == 1:
                        data_indices.append((self.nb + j + 2, self.na + l + 2))
                self.stabilizers.append({'type': 'Z', 'ancilla': anc, 'data': data_indices})

        for i in range(self.ma):
            for j in range(self.nb):
                anc = (j, self.na + i + 2)
                data_indices = []
                for k in range(self.na):
                    if self.Ha[i, k] == 1:
                        data_indices.append((j, k))
                for l in range(self.mb):
                    if self.Hb[l, j] == 1:
                        data_indices.append((self.nb + l + 2, self.na + i + 2))
                self.stabilizers.append({'type': 'X', 'ancilla': anc, 'data': data_indices})
        
        ka = self.na - self.ma
        kb = self.nb - self.mb
        self.logical_Z = [[i * self.nb + j for j in range(kb)] for i in range(ka)]
        self.logical_X = [[j * self.na + i for j in range(kb)] for i in range(ka)]

    def get_info(self):
        # The returned coordinates are all sorted
        return {
            'Hx' : self.Hx,
            'Hz' : self.Hz, 
            'n_data_qubits' : self.n_data_qubits,
            'n_x_stabs' : self.n_x_stabs, 
            'n_z_stabs' : self.n_z_stabs,
            'qubit_coords': self.qubit_coords,
            'data_coords_1': self.data_coords_1,
            'data_coords_2': self.data_coords_2,
            'data_coords': self.data_coords,
            'ancilla_coords': self.ancilla_coords,
            'ancilla_coords_x': self.ancilla_coords_x,
            'ancilla_coords_z': self.ancilla_coords_z,
            'stabilizers': self.stabilizers,
            'coord_to_index': self.coord_to_index,
            'index_to_coord': self.index_to_coord,
            'logical_Z': self.logical_Z,
            'logical_X': self.logical_X
        }
    
    def get_parity_check_matrix(self):
        pass
    
    def build_standard_sm_round(self, noise_profile: list, code_capacity = False) -> stim.Circuit:
        
        p1, p2, p_M, p_R = noise_profile

        # This one-round circuit only contains operations and noise, without detectors
        circuit = stim.Circuit()

        # Before a sm round, insert data qubit errors
        data_indices = [self.coord_to_index[c] for c in self.data_coords]
        circuit.append_operation("DEPOLARIZE1", data_indices, p1)

        # X stabilizers
        Gx = build_tanner_graph(self.Hx, label_prefix='X')
        coloring_x = random_edge_coloring(Gx)

        # TICK 1: Hadamard on X ancillas
        H_targets = [self.coord_to_index[coord] for coord in self.ancilla_coords_x]
        circuit.append_operation("H", H_targets)
        if not code_capacity:
            circuit.append_operation("DEPOLARIZE1", H_targets, p1)
        circuit.append("TICK")

        layer_x = {}
        for (s, q), c in coloring_x.items():
            layer_x.setdefault(c, []).append((int(s[1:]), int(q[1:])))

        for color in sorted(layer_x):
            pair_targets = []
            for s_idx, q_idx in layer_x[color]:
                pair_targets += [self.x_ancilla_offset + s_idx, q_idx]
                circuit.append_operation("CX", [self.x_ancilla_offset + s_idx, q_idx])
            circuit.append_operation("DEPOLARIZE2", pair_targets, p2)
            circuit.append("TICK")

        # TICK: Hadamard on X ancillas
        circuit.append_operation("H", H_targets)
        if not code_capacity:
            circuit.append_operation("DEPOLARIZE1", H_targets, p1)
        circuit.append("TICK")

        # Z stabilizers
        Gz = build_tanner_graph(self.Hz, label_prefix='Z')
        coloring_z = random_edge_coloring(Gz)

        layer_z = {}
        for (s, q), c in coloring_z.items():
            layer_z.setdefault(c, []).append((int(s[1:]), int(q[1:])))

        for color in sorted(layer_z):
            pair_targets = []
            for s_idx, q_idx in layer_z[color]:
                pair_targets += [q_idx, self.z_ancilla_offset + s_idx]
                circuit.append_operation("CX", [q_idx, self.z_ancilla_offset + s_idx])
            circuit.append_operation("DEPOLARIZE2", pair_targets, p2)
            circuit.append("TICK")

        # TICK: MR with SPAM noise
        measure_targets = [self.coord_to_index[coord] for coord in self.ancilla_coords]
        if not code_capacity:
            circuit.append_operation("X_ERROR", measure_targets, p_M)
        circuit.append_operation("MR", measure_targets)
        if not code_capacity:
            circuit.append_operation("X_ERROR", measure_targets, p_R)
        # Don't insert TICK here, insert TICK after specifying the detectors in the construction of the full circuit

        return circuit
    
    def build_full_HGP_code_circuit(self, rounds: int, noise_profile: list, observable_type: str, code_capacity = False) -> stim.Circuit:
    
        p1, p2, p_M, p_R = noise_profile

        full_circuit = stim.Circuit()
        repeat_circuit = stim.Circuit()

        # QUBIT_COORDS annotations
        for coord, index in self.coord_to_index.items():
            full_circuit.append_operation("QUBIT_COORDS", [index], list(coord))

        # Initialization
        data_indices = [self.coord_to_index[c] for c in self.data_coords]
        ancilla_indices = [self.coord_to_index[c] for c in self.ancilla_coords]
        if observable_type == "Z":
            full_circuit.append_operation("R", data_indices)
            if not code_capacity:
                full_circuit.append_operation("X_ERROR", data_indices, p_R)
        else:
            full_circuit.append_operation("RX", data_indices)
            if not code_capacity:
                full_circuit.append_operation("Z_ERROR", data_indices, p_R)
        
        full_circuit.append_operation("R", ancilla_indices)
        if not code_capacity:
            full_circuit.append_operation("X_ERROR", ancilla_indices, p_R)
        full_circuit.append("TICK")

        # First round
        full_circuit += self.build_standard_sm_round(noise_profile)

        # Later rounds
        repeat_circuit.append("TICK")
        repeat_circuit += self.build_standard_sm_round(noise_profile, code_capacity=code_capacity)
        repeat_circuit.append_operation("SHIFT_COORDS", [], [0,0,1])
        # Insert detectors
        for k, ancilla in enumerate(reversed(self.ancilla_coords)):
            # ancilla_index = self.coord_to_index[ancilla]
            prev = -k - 1 - len(self.ancilla_coords)
            repeat_circuit.append_operation("DETECTOR", [stim.target_rec(-k - 1), stim.target_rec(prev)], list(ancilla) + [0])

        full_circuit += repeat_circuit * (rounds-1)

        # Final measurement of data qubits in Z or X basis
        if observable_type == "X":
            if not code_capacity:
                full_circuit.append_operation("Z_ERROR", data_indices, p_M)
            full_circuit.append_operation("MX", data_indices),
        else: # if it's "Z"
            if not code_capacity:
                full_circuit.append_operation("X_ERROR", data_indices, p_M)
            full_circuit.append_operation("M", data_indices)

        # Final detectors
        for stab in self.stabilizers:
            if stab['type'] == observable_type:
                anc = stab['ancilla']
                try:
                    ancilla_index = self.ancilla_coords[::-1].index(anc)
                    anc_rec = stim.target_rec(-ancilla_index - 1 - len(self.data_coords))
                    data_rec_targets = [stim.target_rec(-self.data_coords[::-1].index(q) - 1) for q in stab['data']]
                    full_circuit.append_operation("DETECTOR", [anc_rec] + data_rec_targets, list(anc) + [1])
                except ValueError:
                    continue
        
        # logical observables
        logical_key = 'logical_' + observable_type
        for l, logical_data in enumerate(getattr(self, logical_key)):
            data_indices = [self.data_coords[::-1].index(self.index_to_coord[data]) for data in logical_data]
            full_circuit.append_operation("OBSERVABLE_INCLUDE", [stim.target_rec(-k-1) for k in data_indices], l)

        return full_circuit
    
    @staticmethod
    def sort_coords(coords: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return sorted(coords, key=lambda c: (c[1], c[0]))
    

def build_tanner_graph(H, label_prefix='X'):
    """
    Convert parity-check matrix H into a bipartite Tanner graph.
    Nodes: stabilizers ('X0', 'X1', ...) and qubits ('q0', 'q1', ...)
    Edges: stabilizer - qubit if H[i, j] == 1
    """
    G = nx.Graph()
    n_rows, n_cols = H.shape

    # Add nodes
    for i in range(n_rows):
        G.add_node(f'{label_prefix}{i}', bipartite=0)
    for j in range(n_cols):
        G.add_node(f'q{j}', bipartite=1)

    # Add edges
    for i in range(n_rows):
        for j in range(n_cols):
            if H[i, j] == 1:
                G.add_edge(f'{label_prefix}{i}', f'q{j}')

    return G

def random_edge_coloring(G):
    """
    Assign colors (integers) to edges so that no two edges incident to the same node have the same color.
    Simple greedy strategy with random edge order.
    """
    edges = list(G.edges())
    random.shuffle(edges)

    edge_colors = {}
    node_edge_colors = {node: set() for node in G.nodes()}

    for edge in edges:
        u, v = edge
        used_colors = node_edge_colors[u].union(node_edge_colors[v])
        color = 0
        while color in used_colors:
            color += 1
        edge_colors[edge] = color
        node_edge_colors[u].add(color)
        node_edge_colors[v].add(color)

    return edge_colors    
    
import networkx as nx
import torch
import torch.optim as optim
import logging
from quantum.qaoa import QuantaCoreQAOA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_logistics_graph():
    """Creates a sample conflict graph representing logistics/routing nodes."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    return graph

def main():
    logger = logging.getLogger(__name__)
    logger.info("Initializing QuantaCore QAOA Solver...")

    problem_graph = create_logistics_graph()
    qaoa_depth = 3
    
    qaoa_engine = QuantaCoreQAOA(graph=problem_graph, depth=qaoa_depth)
    quantum_circuit = qaoa_engine.build_circuit()

    # Gammas (cost) and Betas (mixer) initialized for optimization
    params = torch.rand((2, qaoa_depth), requires_grad=True)
    optimizer = optim.Adam([params], lr=0.1)

    logger.info("Starting Hybrid Optimization Loop...")
    for i in range(50):
        optimizer.zero_grad()
        loss = quantum_circuit(params)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            logger.info(f"Optimization Step {i + 1:02d} | Cost Value: {loss.item():.4f}")

    logger.info("Optimization Complete.")

if __name__ == "__main__":
    main()

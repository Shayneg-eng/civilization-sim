# Civilization Sim

An agent-based simulation of a small civilization built on a network of interacting "neuron"
agents, with configurable dynamics and convergence analysis.

## Files
| File | Role |
|---|---|
| `main.py` | Entry point — runs the simulation |
| `config.py` | Simulation parameters |
| `network.py` | The agent/interaction network |
| `neuron.py` | Individual agent behavior |
| `convergence.py` | Measures whether/how the system settles |

## Run it
```bash
python -m pip install -r requirements.txt
python main.py
```

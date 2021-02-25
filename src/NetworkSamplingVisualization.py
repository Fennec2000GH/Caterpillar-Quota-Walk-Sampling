
from mesa import Agent, Model
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

def agent_portrayal(agent: Agent):
    portrayal = {
                    "Shape": "Circle",
                    "Filled": "True",
                    "Layer": "0".
                    "Color": "red"
                }
    return portrayal

grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)
server = ModularServer()





AI AGENT BASED MODEL - Shimmering

# ¿Qué mejoras se incorporaron?
    
    ✅ Decisión de pánico basada en red neuronal (MLP).
    ✅ Factores individuales como sensibilidad y cantidad de abejas vecinas.
    ✅ Cada abeja decide por sí misma en lugar de seguir reglas fijas.

Este modelo hace que cada abeja sea un agente más realista e inteligente, permitiéndole aprender a reaccionar de manera más natural a los depredadores.

## MLP

Vamos a implementar un Perceptrón Multi capa (MLP) en cada abeja para que decida si entra en pánico en función de las condiciones externas. Esto permitirá que cada abeja aprenda patrones de peligro de manera más adaptativa. 😃🐝
Enfoque del modelo

## Entrada de la red neuronal:

    - Distancia al depredador.
    - Número de abejas cercanas que están huyendo.
    - Sensibilidad individual de cada abeja.

## Capas del modelo:
    
    - Capa oculta con función de activación ReLU.
    - Capa de salida con activación sigmoide (probabilidad de entrar en pánico).

## Proceso: Cada abeja pasará estos valores a su MLP para decidir si huye o no. Vamos a usar PyTorch para la red neuronal.




# CODIGO
    
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mesa import Agent, Model
    from mesa.time import RandomActivation
    from mesa.space import ContinuousSpace
    
    # Red neuronal MLP para decidir el pánico
    class BeeNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3, 8)  # 3 entradas -> 8 neuronas ocultas
            self.fc2 = nn.Linear(8, 1)  # 8 neuronas ocultas -> 1 salida
            self.sigmoid = nn.Sigmoid()
    
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.sigmoid(self.fc2(x))
            return x
    
    class Bee(Agent):
        def __init__(self, unique_id, model):
            super().__init__(unique_id, model)
            self.nn = BeeNN()
            self.fleeing = False
            self.steps_fleeing = 0
            self.flee_duration = np.random.randint(3, 7)
    
        def step(self):
            # Distancia al depredador
            predator_distances = [self.model.space.get_distance(self.pos, p.pos) for p in self.model.predators]
            min_predator_distance = min(predator_distances)
    
            # Cantidad de abejas cercanas que están huyendo
            fleeing_neighbors = sum(1 for bee in self.model.schedule.agents
                                    if isinstance(bee, Bee) and bee.fleeing and self.model.space.get_distance(self.pos, bee.pos) < 4)
    
            # Sensibilidad aleatoria de cada abeja (factor de predisposición)
            sensitivity = np.random.uniform(0.5, 1.5)
    
            # Entrada a la red neuronal
            input_tensor = torch.tensor([min_predator_distance, fleeing_neighbors, sensitivity], dtype=torch.float32)
            flee_prob = self.nn(input_tensor).item()
    
            if flee_prob > 0.6:  # Umbral de decisión basado en red neuronal
                self.fleeing = True
    
            if self.fleeing:
                if self.steps_fleeing < self.flee_duration:
                    self.model.space.move_agent(self, (self.pos[0] + np.random.uniform(-1.5, 1.5), 
                                                       self.pos[1] + np.random.uniform(-1.5, 1.5)))
                    self.steps_fleeing += 1
                else:
                    self.fleeing = False
                    self.steps_fleeing = 0
    
    class Predator(Agent):
        def __init__(self, unique_id, model, pos):
            super().__init__(unique_id, model)
            self.pos = pos
    
        def step(self):
            pass  # Depredador estático
    
    class BeeSwarmModel(Model):
        def __init__(self, num_bees, predator_pos):
            self.space = ContinuousSpace(20, 20, torus=False)
            self.schedule = RandomActivation(self)
            self.predators = []
    
            for i in range(num_bees):
                bee = Bee(i, self)
                x, y = np.random.uniform(8, 12), np.random.uniform(8, 12)
                self.space.place_agent(bee, (x, y))
                self.schedule.add(bee)
    
            predator = Predator(num_bees, self, predator_pos)
            self.space.place_agent(predator, predator_pos)
            self.predators.append(predator)
            self.schedule.add(predator)
    
        def step(self):
            self.schedule.step()
    
    # Simulación y animación
    model = BeeSwarmModel(num_bees=50, predator_pos=(10, 5))
    fig, ax = plt.subplots()
    
    def update(frame):
        model.step()
        ax.clear()
        x_bees = [agent.pos[0] for agent in model.schedule.agents if isinstance(agent, Bee)]
        y_bees = [agent.pos[1] for agent in model.schedule.agents if isinstance(agent, Bee)]
        ax.scatter(x_bees, y_bees, color='yellow', label='Abejas')
        ax.scatter(*model.predators[0].pos, color='red', marker='X', label='Depredador')
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.legend()
    
    ani = animation.FuncAnimation(fig, update, frames=20, interval=500)
    plt.show()

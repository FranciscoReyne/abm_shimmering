
## Mejoras implementadas
  
    ✅ Cada abeja decide si inicia el shimmering usando una red neuronal (MLP).
    ✅ El efecto se propaga en cadena, activándose en abejas cercanas.
    ✅ Visualización con colores, mostrando qué abejas están activas en el shimmering.
    ✅ Duración variable del efecto, haciendo que no todas las abejas actúen al mismo tiempo.

## AGENT BASED MODEL - Shimmering
    
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mesa import Agent, Model
    from mesa.time import RandomActivation
    from mesa.space import ContinuousSpace
    
    # Red neuronal MLP para decidir si la abeja inicia el shimmering
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
            self.shimmering = False
            self.steps_shimmering = 0
            self.shimmer_duration = np.random.randint(3, 7)  # Duración variable del efecto
    
        def step(self):
            # Distancia al depredador
            predator_distances = [self.model.space.get_distance(self.pos, p.pos) for p in self.model.predators]
            min_predator_distance = min(predator_distances)
    
            # Cantidad de abejas cercanas que están en shimmering
            shimmering_neighbors = sum(1 for bee in self.model.schedule.agents
                                       if isinstance(bee, Bee) and bee.shimmering and self.model.space.get_distance(self.pos, bee.pos) < 4)
    
            # Sensibilidad individual de cada abeja
            sensitivity = np.random.uniform(0.5, 1.5)
    
            # Entrada a la red neuronal
            input_tensor = torch.tensor([min_predator_distance, shimmering_neighbors, sensitivity], dtype=torch.float32)
            shimmer_prob = self.nn(input_tensor).item()
    
            if shimmer_prob > 0.6:  # Umbral de decisión basado en red neuronal
                self.shimmering = True
    
            if self.shimmering:
                if self.steps_shimmering < self.shimmer_duration:
                    self.steps_shimmering += 1
                else:
                    self.shimmering = False
                    self.steps_shimmering = 0
    
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
                x, y = np.random.uniform(8, 12), np.random.uniform(8,12)
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
        colors = ['gold' if agent.shimmering else 'yellow' for agent in model.schedule.agents if isinstance(agent, Bee)]
        
        ax.scatter(x_bees, y_bees, c=colors, label='Abejas')
        ax.scatter(*model.predators[0].pos, color='red', marker='X', label='Depredador')
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.legend()
    
    ani = animation.FuncAnimation(fig, update, frames=20, interval=500)
    plt.show()

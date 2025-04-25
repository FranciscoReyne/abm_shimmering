# abm_shimmering
Abejas que generan un movimiento coordinado para parecer un organismo más grande y ahuyentar a un depredador.


    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mesa import Agent, Model
    from mesa.time import RandomActivation
    from mesa.space import ContinuousSpace
    
    class Bee(Agent):
        def __init__(self, unique_id, model):
            super().__init__(unique_id, model)
            self.fleeing = False
            self.reaction_time = np.random.randint(1, 4)  # Algunas abejas reaccionan más rápido
            self.flee_duration = np.random.randint(3, 7)  # Tiempo que dura la huida
            self.steps_fleeing = 0
    
        def step(self):
            if self.fleeing:
                if self.steps_fleeing < self.flee_duration:
                    self.model.space.move_agent(self, (self.pos[0] + np.random.uniform(-1.5, 1.5), 
                                                       self.pos[1] + np.random.uniform(-1.5, 1.5)))
                    self.steps_fleeing += 1
                else:
                    self.fleeing = False
                    self.steps_fleeing = 0
    
            else:
                # Si el depredador está cerca o una abeja cercana está huyendo, se activa la reacción
                for predator in self.model.predators:
                    if self.model.space.get_distance(self.pos, predator.pos) < 3:
                        self.fleeing = True
                        break
    
                if not self.fleeing:
                    for bee in self.model.schedule.agents:
                        if isinstance(bee, Bee) and bee.fleeing and self.model.space.get_distance(self.pos, bee.pos) < 4:
                            self.fleeing = True
                            break
    
    class Predator(Agent):
        def __init__(self, unique_id, model, pos):
            super().__init__(unique_id, model)
            self.pos = pos
    
        def step(self):
            pass  # Depredador estático en este modelo básico
    
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

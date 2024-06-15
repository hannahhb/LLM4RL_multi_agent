from .historicalobs import *
from .doorkey import *
from .lavadoorkey import *
from .twodoor import *
from .coloreddoorkey import *
from .lockedhallway import *

gym.envs.register(
    id='MiniGrid-SimpleDoorKey-Min5-Max10-View3',
    entry_point='env.doorkey:DoorKeyEnv',
    kwargs={'minRoomSize' : 5, \
            'maxRoomSize' : 10, \
            'agent_view_size' : 3, \
            'max_steps': 150},
)

gym.envs.register(
    id='MiniGrid-LavaDoorKey-Min5-Max10-View3',
    entry_point='env.lavadoorkey:LavaDoorKeyEnv',
    kwargs={'minRoomSize' : 5, \
            'maxRoomSize' : 10, \
            'agent_view_size' : 3, \
            'max_steps': 150},
)

gym.envs.register(
    id='MiniGrid-ColoredDoorKey-Min5-Max10-View3',
    entry_point='env.coloreddoorkey:ColoredDoorKeyEnv',
    kwargs={'minRoomSize' : 5, \
            'maxRoomSize' : 10, \
            'minNumKeys' : 2, \
            'maxNumKeys' : 2, \
            'agent_view_size' : 3, \
            'max_steps' : 150},
)

gym.envs.register(
    id='MiniGrid-TwoDoor-Min20-Max20',
    entry_point='env.twodoor:TwoDoorEnv',
    kwargs={'minRoomSize' : 20, \
            'maxRoomSize' : 20, \
            'agent_view_size' : 3, \
            'max_steps' : 150},
)

CONFIGURATIONS = {
    'MultiGrid-LockedHallway-2Rooms-v0': (LockedHallwayEnv, {'num_rooms': 2}),
    'MultiGrid-LockedHallway-4Rooms-v0': (LockedHallwayEnv, {'num_rooms': 4}),
    'MultiGrid-LockedHallway-6Rooms-v0': (LockedHallwayEnv, {'num_rooms': 6}),
}

# Register environments with gymnasium
from gymnasium.envs.registration import register
for name, (env_cls, config) in CONFIGURATIONS.items():
    register(id=name, entry_point=env_cls, kwargs=config)
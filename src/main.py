#python time
from ursina import *

app = Ursina()

atom = Entity(
    model="sphere",
    color=color.red,
    scale=1,
    position=(0,1,0),
)

EditorCamera()

if __name__ == "__main__":
    app.run()
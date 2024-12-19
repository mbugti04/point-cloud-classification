import pyglet

window = pyglet.window.Window()

@window.event
def on_draw():
    window.clear()
    # Load and display the 'paris.glb' file
    model = pyglet.resource.model('paris.glb')
    model.draw()

pyglet.app.run()
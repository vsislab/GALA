from math import sin, cos, pi
from functools import partial


def create_box(client, halfExtents, mass=0, friction=1.5, rgba=(0.55, 0.55, 0.55, 1)):
    shape_id = client.createCollisionShape(client.GEOM_BOX, halfExtents=halfExtents)
    body_id = client.createMultiBody(baseMass=mass,
                                     baseCollisionShapeIndex=shape_id)
    texture_id = client.loadTexture("env/assets/terrain/wooden.jpg")
    client.changeVisualShape(body_id, -1, rgbaColor=[1, 1, 1, 0.1], textureUniqueId=texture_id)
    client.changeVisualShape(body_id, -1, rgbaColor=[1.2, 1.2, 1, 1])
    client.changeDynamics(body_id, -1, lateralFriction=friction, restitution=0.)
    return body_id


def create_shelf(env, abs_X, abs_Y):
    l, s, h = 0.75, 0.05, 0.17
    x, y, z = 0.5, 2., h
    env.add_object(partial(create_box, halfExtents=(s, l + s, s)),
                   position=(abs_X, y + abs_Y, z + h))
    env.add_object(partial(create_box, halfExtents=(s, s, h)),
                   position=(abs_X, y + abs_Y + l, z))
    env.add_object(partial(create_box, halfExtents=(s, s, h)),
                   position=(abs_X, y + abs_Y - l, z))


def create_pile(env, abs_X, abs_Y):
    l, s, h = 0.75, 0.05, 0.15
    x, y, z = 0.5, 2., h
    env.add_object(partial(create_box, halfExtents=(s, s, h)),
                   position=(abs_X, y + abs_Y + l, z))
    env.add_object(partial(create_box, halfExtents=(s, s, h)),
                   position=(abs_X, y + abs_Y - l, z))


def create_bridge(env, abs_X, abs_Y):
    theta = 22 * pi / 180
    s1, s2, w, h, d = 2, 0.75, 0.1, 0.01, 0.27
    x, y, z = s1, 1.5, sin(theta) * s2 * 2
    env.add_object(partial(create_box, halfExtents=(s1 - 1, w, h)),  # single bridge
                   position=(0 + abs_X, y + abs_Y, z))

    env.add_object(partial(create_box, halfExtents=(0.5, 0.5, h)),  # two platform
                   position=(0 + abs_X - s1 + 0.5, y + abs_Y, z))
    env.add_object(partial(create_box, halfExtents=(0.5, 0.5, h)),
                   position=(0 + abs_X + s1 - 0.5, y + abs_Y, z))

    env.add_object(partial(create_box, halfExtents=(s1 - 0.85, w, h)),  # double bridge
                   position=(0 + abs_X + s1 * 2 - 0.9, y + abs_Y - d - 0.05, z))
    env.add_object(partial(create_box, halfExtents=(s1 - 0.85, w, h)),
                   position=(0 + abs_X + s1 * 2 - 0.9, y + abs_Y + d - 0.05, z))

    env.add_object(partial(create_box, halfExtents=(0.5, 0.5, h)),  # platform
                   position=(0 + abs_X + s1 * 2 + 0.5, y + abs_Y, z))

    env.add_object(partial(create_box, halfExtents=(s2, 0.5, h)),  # two slop
                   position=(x + abs_X + cos(theta) * s2 - sin(theta) * h + s1 + 1, y + abs_Y, z - sin(theta) * s2 + h * (1 - cos(theta))),
                   orientation=env.client.getQuaternionFromEuler((0, theta, 0)))

    env.add_object(partial(create_box, halfExtents=(s2, 0.5, h)),
                   position=(-x + abs_X - cos(theta) * s2 + sin(theta) * h, y + abs_Y, z - sin(theta) * s2 + h * (1 - cos(theta))),
                   orientation=env.client.getQuaternionFromEuler((0, -theta, 0)))


def create_bar(env, abs_X, abs_Y, H):
    l, w, h = 1, 0.05, H
    x, y, z = 0, 2, h
    env.add_object(partial(create_box, halfExtents=(l, w, h)),
                   position=(0 + abs_X, y + abs_Y, z))


def create_integrate(env):
    d = 4.5
    create_bridge(env, d - 5.5, 5)

    create_shelf(env, d, 6.6)
    create_shelf(env, d - 0.5, 6.6)
    create_shelf(env, d - 1, 6.6)
    create_shelf(env, d - 1.5, 6.6)

    # create_bar(env, d - 3, 6.85, 0.1)
    create_bar(env, d - 4.3, 6.65, 0.17)

    create_pile(env, d - 7, 6.7)

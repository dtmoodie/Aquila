import metaobject as mo
import glob

plugin = glob.glob('/home/dan/code/EagleEye/Aquila/dependencies/MetaObject_build/bin/plugins/libmo_objectplugin*')
assert(mo.plugins.loadPlugin(plugin[0]))
print(mo.listConstructableObjects())

obj = mo.object.SerializableObject()

assert(obj.test == 5)
assert(obj.test2 == 6)

obj.test = 10
obj.test2 = 20

assert(obj.test == 10)
assert(obj.test2 == 20)

obj = mo.object.DerivedSignals()

assert(obj.base_param == 5)

types = mo.listConstructableObjects()

pt = mo.datatypes.Point2d(x=1, y=2)

assert(pt.x == 1.0)
assert(pt.y == 2.0)
pt.x = 2.0
pt.y = 3.0
assert(pt.x == 2.0)
assert(pt.y == 3.0)

print('Success')

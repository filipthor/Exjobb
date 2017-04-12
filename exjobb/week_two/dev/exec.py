from week_two.dev import source


room = source.Room(source.Wall("North",30),source.Wall("West",30),source.Wall("South",15),source.Wall("East",15))

solver = source.Solver(problem=room)

solver.run()
solver.visualize()
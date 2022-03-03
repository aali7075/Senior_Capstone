
class FieldNuller:
    def __init__(self, shape, coil_size, coil_spacing, wall_spacing, max_current):
        self.shape = shape
        self.coil_size = coil_size
        self.coil_spacing = coil_spacing
        self.wall_spacing = wall_spacing
        self.max_current = max_current

        self.coords = self.__get_coordinates()
        self.b = self.__get_b()


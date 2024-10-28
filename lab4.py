
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    @property
    def name(self):
        return self.name

    def age(self):
        return self.age

    @name.setter
    def name(self, name):
        self.name = name

    @age.setter
    def age(self, age):
        if age < 0:
            raise ValueError("Age must be greater than or equal to 0")
        self.age = age

    def add_grade(self, grade):
        self.grade = grade
        return self.grade
    def calculate_average(self):
        sum = 0
        for grade in self.grade:
            sum += grade
        return sum / len(self.grade)


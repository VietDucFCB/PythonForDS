import numpy as np

class Student:
    def __init__(self, name, age):
        self._name = name
        self._age = age
        self.grades = []

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, age):
        if age < 0:
            raise ValueError("Age must be greater than or equal to 0")
        self._age = age

    def add_grade(self, grade):
        """Thêm điểm vào danh sách điểm"""
        if 0 <= grade <= 100:
            self.grades.append(grade)
        else:
            raise ValueError("Grade must be between 0 and 100")

    def calculate_average(self):
        """Tính điểm trung bình của học sinh"""
        if not self.grades:
            return 0
        return sum(self.grades) / len(self.grades)

    def describe(self):
        """Mô tả thông tin của học sinh"""
        avg_grade = self.calculate_average()
        return f"Name: {self.name}, Age: {self.age}, Average Grade: {avg_grade:.2f}"


class School:
    def __init__(self):
        self.students = []

    def add_student(self, student):
        """Thêm học sinh vào danh sách"""
        self.students.append(student)

    def remove_student(self, name):
        """Xóa học sinh khỏi danh sách theo tên"""
        self.students = [s for s in self.students if s.name != name]

    def find_student(self, name):
        """Tìm kiếm học sinh theo tên"""
        for student in self.students:
            if student.name == name:
                return student.describe()
        return "Student not found"

    def get_top_student(self):
        """Trả về học sinh có điểm trung bình cao nhất"""
        if not self.students:
            return "No students in the school"
        top_student = max(self.students, key=lambda s: s.calculate_average())
        return top_student.describe()

school = School()

# Thêm học sinh vào trường
for i in range(5):
    student = Student(name=f"Student {i+1}", age=np.random.randint(15, 19))
    for _ in range(5):  # Thêm 5 điểm cho mỗi học sinh
        student.add_grade(np.random.uniform(0, 100))
    school.add_student(student)

# Tìm và hiển thị học sinh theo tên
print(school.find_student("Student 3"))

# Hiển thị học sinh có điểm trung bình cao nhất
print(school.get_top_student())

class Student:
    def __init__(self, name, age):
        self._name = name  # Thuộc tính name là private
        self._age = age    # Thuộc tính age là private
        self.grades = []  # Danh sách điểm của học sinh
        self._average = 0  # Điểm trung bình

    # Getter cho name
    @property
    def name(self):
        return self._name

    # Setter cho name
    @name.setter
    def name(self, name):
        self._name = name

    # Getter cho age
    @property
    def age(self):
        return self._age

    # Setter cho age
    @age.setter
    def age(self, age):
        self._age = age

    # Thêm điểm vào danh sách điểm
    def add_grade(self, grade):
        self.grades.append(grade)
        self._calculate_average()  # Cập nhật điểm trung bình sau khi thêm điểm

    # Tính và cập nhật điểm trung bình
    def _calculate_average(self):
        if len(self.grades) > 0:
            self._average = sum(self.grades) / len(self.grades)

    # Trả về điểm trung bình của học sinh
    @property
    def average(self):
        return self._average

    # Mô tả thông tin của học sinh
    def describe(self):
        return f"{self._name}, tuổi {self._age}, điểm trung bình: {self._average:.2f}"


class School:
    def __init__(self):
        self.students = []  # Danh sách học sinh trong trường

    # Thêm học sinh vào trường
    def add_student(self, student):
        self.students.append(student)
        print(f"Đã thêm học sinh {student.name} vào trường.")

    # Xóa học sinh khỏi trường theo tên
    def remove_student(self, name):
        for student in self.students:
            if student.name == name:
                self.students.remove(student)
                print(f"Đã xóa học sinh {name} khỏi trường.")
                return
        print(f"Không tìm thấy học sinh có tên {name}.")

    # Tìm học sinh theo tên
    def find_student(self, name):
        for student in self.students:
            if student.name == name:
                return student.describe()
        return f"Không tìm thấy học sinh có tên {name}."

    # Trả về học sinh có điểm trung bình cao nhất
    def get_top_student(self):
        if not self.students:
            return "Không có học sinh trong danh sách."
        top_student = max(self.students, key=lambda student: student.average)
        return f"Học sinh có điểm cao nhất là {top_student.name} với điểm trung bình: {top_student.average:.2f}"


# Test chương trình
school = School()

# Tạo học sinh
student1 = Student("An", 16)
student2 = Student("Bình", 15)
student3 = Student("Cường", 17)

# Thêm điểm cho học sinh
student1.add_grade(9)
student1.add_grade(9)

student2.add_grade(7)
student2.add_grade(6)

student3.add_grade(10)
student3.add_grade(9)

# Thêm học sinh vào trường
school.add_student(student1)
school.add_student(student2)
school.add_student(student3)

# Hiển thị học sinh có điểm cao nhất
print(school.get_top_student())

# Tìm học sinh theo tên
print(school.find_student("Bình"))

# Xóa học sinh
school.remove_student("An")
print(school.find_student("An"))
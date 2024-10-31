from itertools import combinations_with_replacement


def generate_candy_distributions():
    # Số viên kẹo cần thêm để mỗi hộp có ít nhất 3 viên kẹo
    remaining_candies = 4
    # Tìm tất cả các tổ hợp (y_1, y_2, y_3, y_4) sao cho y_1 + y_2 + y_3 + y_4 = 4
    solutions = []

    for comb in combinations_with_replacement(range(remaining_candies + 1), 4):
        if sum(comb) == remaining_candies:
            # Chuyển đổi từ y_i sang x_i
            distribution = [y + 3 for y in comb]
            solutions.append(distribution)

    return solutions

# Hiển thị tất cả các cách sắp xếp
distributions = generate_candy_distributions()
for i, distribution in enumerate(distributions, start=1):
    print(f"Cách {i}: {distribution}")

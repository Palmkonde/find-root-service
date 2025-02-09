import matplotlib.pyplot as plt
import numpy as np
import sympy
import sys
import matplotlib

from flask import Flask, render_template
from flask import request
from typing import Tuple, List
from PIL import Image


class FindRoot:
    def __init__(self,
                 function: str,
                 range_input: Tuple[int, int],
                 epsilon: float) -> None:
        self.function = sympy.sympify(function)
        self.start = range_input[0]
        self.end = range_input[1]
        self.epsilon = epsilon

        self.x = sympy.symbols('x')

    def bisection(self) -> Tuple[float, int, List[float]]:
        print("Starting Bisection")
        x_vals = []
        a = self.start
        b = self.end
        mid_point = None
        it = 0

        while abs(b - a) > self.epsilon:
            mid_point = a + (b - a) / 2
            f_m = self.function.subs(self.x, mid_point)
            f_a = self.function.subs(self.x, a)
            f_b = self.function.subs(self.x, b)

            print(mid_point)

            x_vals.append(mid_point)
            it += 1
            if f_m == 0:
                break
            elif f_a * f_m < 0:
                b = mid_point
            else:
                a = mid_point

        print("End bisection")
        return mid_point, it, x_vals

    def newton_method(self) -> Tuple[float, int, List[float]]:
        print("Starting newton")
        df = sympy.diff(self.function, self.x)
        x_0 = np.random.randint(self.start, self.end)
        prev_x = None
        x_vals = []
        it = 0

        while True:
            f_x = self.function.subs(self.x, sympy.Rational(x_0)).evalf()
            df_x = df.subs(self.x, sympy.Rational(x_0)).evalf()

            x_vals.append(x_0)
            print(x_0)
            if abs(df_x) <= 1e-2:
                break

            x_0 = x_0 - f_x/df_x

            it += 1

            if prev_x and abs(x_0 - prev_x) < self.epsilon:
                break
            prev_x = x_0

        print("End newton")
        return x_0, it, x_vals

    def find_roots_in_range(self) -> List[int | float]:
        roots = sympy.solve(self.function, self.x)
        root_in_range = []

        for root in roots:
            if not root.is_real:
                continue

            if self.start <= root <= self.end:
                root_in_range.append(root)
        print(root_in_range)
        return root_in_range


app = Flask(__name__)


def find_nearest_from_list(root: float, real_solutions: List[float]) -> float:
    nearest_root = None
    min_dist = float('inf')

    for real_sol in real_solutions:
        dist = abs(root - real_sol)
        if dist < min_dist:
            min_dist = dist
            nearest_root = root

    return nearest_root


def plot_base_graph(f_x, range_input: Tuple[int, int]) -> Tuple[List[float], List[float]]:
    x = sympy.symbols('x')
    f_x = sympy.simplify(f_x)

    x_list = np.linspace(range_input[0], range_input[1])
    y = [f_x.subs(x, x_vals) for x_vals in x_list]

    return x_list, y


def plot_animation(ax, iter_num, x_val, f_x, name):
    x = sympy.symbols('x')
    f_x = sympy.simplify(f_x)
    temporary_image = r"./tmp_image/"
    image_path = r"./static/image"

    ax.axhline(0, color="black", lw=1)
    ax.set_title(f"{name}")

    for i in range(iter_num):
        ax.plot(x_val[i], f_x.subs(x, x_val[i]), 'ro')
        plt.savefig(f"{temporary_image}/temp_{i}.png")

    frames = []
    for i in range(iter_num):
        img = Image.open(f"{temporary_image}/temp_{i}.png")
        frames.append(img)

    frames[0].save(f"{image_path}/{name}.gif", save_all=True,
                   append_images=frames[1:], loop=1, duration=1000)


@app.route('/')
def home_page() -> None:
    return render_template("index.html")


@app.route("/find_root", methods=["GET", "POST"])
def find_root() -> None:
    if request.method == "POST":
        f_x = request.form["function"]
        from_x = request.form["from-x"]
        to_x = request.form["to-x"]
        epsilon = request.form["error"]

        fr = FindRoot(
            function=f_x,
            range_input=(int(from_x), int(to_x)),
            epsilon=float(epsilon)
        )

        # data
        bisection_answer, it1, x_val1 = fr.bisection()
        newton_answer, it2, x_val2 = fr.newton_method()
        real_solution = fr.find_roots_in_range()
        
        if not real_solution:
            return render_template("index.html", noSol=True)

        bisection_approch = find_nearest_from_list(
            bisection_answer, real_solution)
        newton_approch = find_nearest_from_list(newton_answer, real_solution)

        iter_nums = [i for i in range(max(it1, it2))]
        error_value1 = [abs(x - bisection_approch) for x in x_val1]
        error_value2 = [abs(x - newton_approch) for x in x_val2]

        while len(error_value1) < len(iter_nums):
            error_value1.append(error_value1[-1])

        while len(error_value2) < len(iter_nums):
            error_value2.append(error_value2[-1])

        print("ploting...")

        # plotting Error
        image_path = r"./static/image/Error.png"
        fig, ax = plt.subplots()
        ax.set_title("Iteration vs Error")
        ax.plot(iter_nums, error_value1, label="Bisection")
        ax.plot(iter_nums, error_value2, label="Newton's method")
        ax.set_yscale('log')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error")
        ax.set_xlim(0, max(iter_nums))
        ax.legend()
        plt.savefig(image_path)
        plt.close()

        # plotting bisection
        bi_image = "./static/image/Bisection.gif"
        fig, ax = plt.subplots()
        x_tmp, y_tmp = plot_base_graph(f_x, (int(from_x), int(to_x)))
        ax.plot(x_tmp, y_tmp)
        plot_animation(ax, it1, x_val1, f_x, "Bisection")
        plt.close()
        
        newton_image = "./static/image/Newton.gif"
        fig, ax = plt.subplots()
        x_tmp, y_tmp = plot_base_graph(f_x, (int(from_x), int(to_x)))
        ax.plot(x_tmp, y_tmp)
        plot_animation(ax, it2, x_val2, f_x, "Newton")

    return render_template("index.html", 
                           error_image=image_path,
                           bisection_image=bi_image,
                           newton_image=newton_image)


if __name__ == "__main__":
    matplotlib.use('Agg')
    sys.set_int_max_str_digits(1000)
    app.run(debug=True)

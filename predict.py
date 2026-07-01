import joblib
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path


MODEL_PATH = Path(__file__).with_name("titanic_model.pkl")

# Exact columns used during training
COLUMNS = [
    'Age', 'Fare', 'Sex', 'sibsp', 'zero', 'zero.1', 'zero.2', 'zero.3',
    'zero.4', 'zero.5', 'zero.6', 'Parch', 'zero.7', 'zero.8', 'zero.9',
    'zero.10', 'zero.11', 'zero.12', 'zero.13', 'zero.14', 'Pclass',
    'zero.15', 'zero.16', 'Embarked', 'zero.17', 'zero.18'
]


def load_model():
    return joblib.load(MODEL_PATH)


def build_input_frame(values):
    data = pd.DataFrame([[0] * len(COLUMNS)], columns=COLUMNS)
    data['Age'] = float(values['Age'])
    data['Fare'] = float(values['Fare'])
    data['Sex'] = 1 if values['Sex'] == 'male' else 0
    data['Embarked'] = {'S': 2, 'C': 0, 'Q': 1}[values['Embarked']]
    data['Pclass'] = int(values['Pclass'])
    data['sibsp'] = int(values['sibsp'])
    data['Parch'] = int(values['Parch'])
    return data


class SurvivalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Titanic Survival Predictor")
        self.root.geometry("420x420")
        self.root.resizable(False, False)

        self.model = load_model()

        tk.Label(root, text="Titanic Survival Predictor", font=("Segoe UI", 16, "bold")).pack(pady=(12, 8))
        tk.Label(root, text="Enter passenger details to predict survival.", font=("Segoe UI", 10)).pack()

        self.entries = {}
        self.create_fields()

        button_frame = tk.Frame(root)
        button_frame.pack(pady=12)
        tk.Button(button_frame, text="Predict", width=12, command=self.predict).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Reset", width=12, command=self.reset).pack(side=tk.LEFT, padx=5)

        self.result_var = tk.StringVar(value="Prediction will appear here")
        tk.Label(root, textvariable=self.result_var, font=("Segoe UI", 12, "bold"), wraplength=360, justify="center").pack(pady=12)

    def create_fields(self):
        form = ttk.Frame(self.root, padding=10)
        form.pack(fill="x")

        fields = [
            ("Age", "age", 22.0),
            ("Fare", "fare", 7.25),
            ("Pclass", "pclass", 3),
            ("Siblings/Spouse", "sibsp", 1),
            ("Parents/Children", "parch", 0),
        ]

        for label, key, default in fields:
            row = ttk.Frame(form)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label + ":", width=18).pack(side=tk.LEFT)
            entry = ttk.Entry(row)
            entry.insert(0, str(default))
            entry.pack(side=tk.RIGHT, expand=True, fill="x")
            self.entries[key] = entry

        sex_row = ttk.Frame(form)
        sex_row.pack(fill="x", pady=3)
        ttk.Label(sex_row, text="Sex:", width=18).pack(side=tk.LEFT)
        self.entries['sex'] = ttk.Combobox(sex_row, state="readonly", values=["male", "female"])
        self.entries['sex'].current(0)
        self.entries['sex'].pack(side=tk.RIGHT, expand=True, fill="x")

        embark_row = ttk.Frame(form)
        embark_row.pack(fill="x", pady=3)
        ttk.Label(embark_row, text="Embarked:", width=18).pack(side=tk.LEFT)
        self.entries['embarked'] = ttk.Combobox(embark_row, state="readonly", values=["S", "C", "Q"])
        self.entries['embarked'].current(0)
        self.entries['embarked'].pack(side=tk.RIGHT, expand=True, fill="x")

    def predict(self):
        try:
            values = {
                'Age': self.entries['age'].get(),
                'Fare': self.entries['fare'].get(),
                'Pclass': self.entries['pclass'].get(),
                'sibsp': self.entries['sibsp'].get(),
                'Parch': self.entries['parch'].get(),
                'Sex': self.entries['sex'].get(),
                'Embarked': self.entries['embarked'].get(),
            }
            input_frame = build_input_frame(values)
            prediction = self.model.predict(input_frame)[0]
            result = "Survived" if prediction == 1 else "Did not survive"
            self.result_var.set(f"Prediction: {result}")
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numeric values for Age, Fare, Pclass, Siblings/Spouse, and Parents/Children.")
        except Exception as exc:
            messagebox.showerror("Prediction error", f"Unable to make a prediction: {exc}")

    def reset(self):
        self.entries['age'].delete(0, tk.END)
        self.entries['age'].insert(0, "22")
        self.entries['fare'].delete(0, tk.END)
        self.entries['fare'].insert(0, "7.25")
        self.entries['pclass'].delete(0, tk.END)
        self.entries['pclass'].insert(0, "3")
        self.entries['sibsp'].delete(0, tk.END)
        self.entries['sibsp'].insert(0, "1")
        self.entries['parch'].delete(0, tk.END)
        self.entries['parch'].insert(0, "0")
        self.entries['sex'].current(0)
        self.entries['embarked'].current(0)
        self.result_var.set("Prediction will appear here")


if __name__ == "__main__":
    root = tk.Tk()
    app = SurvivalApp(root)
    root.mainloop()
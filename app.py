import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF


DRILL_TYPES = [
    "Hydraulic Drill Machine",
    "Underground Mining Jumbo Drill",
    "Heavy-Duty Rock Drill",
    "Rotary Drill",
    "Blast Hole Production Drill",
    "Light Drilling Machine",
    "Quarry Drilling Machine",
    "Core Drilling Machine",
    "Diamond Core Drill",
    "Exploration Drill",
    "DTH (Down-The-Hole) Drill",
    "Top Hammer Drill",
]

DRILL_TYPE_FACTORS = {
    "Hydraulic Drill Machine": 1.00,
    "Underground Mining Jumbo Drill": 0.96,
    "Heavy-Duty Rock Drill": 1.02,
    "Rotary Drill": 0.98,
    "Blast Hole Production Drill": 1.05,
    "Light Drilling Machine": 0.90,
    "Quarry Drilling Machine": 0.97,
    "Core Drilling Machine": 0.92,
    "Diamond Core Drill": 0.94,
    "Exploration Drill": 0.95,
    "DTH (Down-The-Hole) Drill": 1.03,
    "Top Hammer Drill": 0.99,
}

BIT_TYPES = [
    "Tungsten Carbide Bit",
    "Button Bit",
    "Cross Bit",
    "Chisel Bit",
    "DTH Bit",
    "Tricone Bit",
    "PDC Bit",
    "Diamond Bit",
    "Core Drill Bit",
]

BIT_LIFE_DB = {
    "Tungsten Carbide Bit": (100, 150),
    "Button Bit": (150, 250),
    "Cross Bit": (80, 120),
    "Chisel Bit": (80, 120),
    "DTH Bit": (200, 350),
    "Tricone Bit": (300, 600),
    "PDC Bit": (500, 1000),
    "Diamond Bit": (1000, 3000),
    "Core Drill Bit": (300, 800),
}

RANGE_DATABASE = {
    "pressure": [(120, 180, "Poor"), (180, 220, "Fair"), (220, 280, "Good"), (280, 350, "Excellent")],
    "flow": [(0, 50, "Poor"), (50, 80, "Fair"), (80, 120, "Good"), (120, 180, "Excellent")],
    "rpm": [(0, 100, "Poor"), (100, 180, "Fair"), (180, 300, "Good"), (300, 450, "Excellent")],
    "temperature": [(-999, 20, "Cold Start"), (20, 40, "Good"), (40, 60, "Excellent"), (60, 75, "Fair"), (75, 90, "Poor")],
}

HISTORY_FILE = "analysis_history.json"


@dataclass
class DrillInputs:
    drill_type: str
    pressure: float
    flow_rate: float
    rpm: float
    temperature: float
    piston_diameter: float
    bit_diameter: float
    bit_type: str
    machine_efficiency: float


class HydraulicDrillAnalyzer:
    """Performs all engineering calculations and diagnostics for hydraulic drill performance."""

    def __init__(self, inputs: DrillInputs):
        self.inputs = inputs
        self.validate_inputs()
        self.pressure_pa = self.inputs.pressure * 1e5

    def validate_inputs(self):
        if not (0 < self.inputs.pressure <= 500):
            raise ValueError("Pressure must be between 0 and 500 bar.")
        if not (0 < self.inputs.flow_rate <= 300):
            raise ValueError("Flow rate must be between 0 and 300 L/min.")
        if not (0 < self.inputs.rpm <= 800):
            raise ValueError("RPM must be between 0 and 800.")
        if not (-20 <= self.inputs.temperature <= 130):
            raise ValueError("Temperature must be between -20 and 130 °C.")
        if not (5 <= self.inputs.piston_diameter <= 500):
            raise ValueError("Piston diameter must be between 5 and 500 mm.")
        if not (5 <= self.inputs.bit_diameter <= 250):
            raise ValueError("Bit diameter must be between 5 and 250 mm.")
        if not (0 <= self.inputs.machine_efficiency <= 100):
            raise ValueError("Machine efficiency must be between 0 and 100%.")

    def hydraulic_power_kw(self) -> float:
        return round((self.inputs.pressure * self.inputs.flow_rate) / 600.0, 2)

    def piston_area_m2(self) -> float:
        radius_m = (self.inputs.piston_diameter / 1000.0) / 2.0
        return round(math.pi * radius_m ** 2, 6)

    def hydraulic_force_n(self) -> float:
        return round(self.pressure_pa * self.piston_area_m2(), 2)

    def _normalize(self, value: float, minimum: float, maximum: float) -> float:
        return max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))

    def _pressure_score(self) -> float:
        return self._normalize(self.inputs.pressure, 120, 350) * 100

    def _flow_score(self) -> float:
        return self._normalize(self.inputs.flow_rate, 50, 180) * 100

    def _rpm_score(self) -> float:
        return self._normalize(self.inputs.rpm, 100, 450) * 100

    def _temperature_score(self) -> float:
        if 40 <= self.inputs.temperature <= 60:
            return 100.0
        if self.inputs.temperature < 40:
            return max(0.0, 100.0 - (40.0 - self.inputs.temperature) * 2.0)
        return max(0.0, 100.0 - (self.inputs.temperature - 60.0) * 2.5)

    def _diameter_score(self, value: float, minimum: float, maximum: float) -> float:
        return self._normalize(value, minimum, maximum) * 100

    def _pressure_factor(self) -> float:
        return self._pressure_score() * 0.25

    def _flow_factor(self) -> float:
        return self._flow_score() * 0.20

    def _rpm_factor(self) -> float:
        return self._rpm_score() * 0.20

    def _temperature_factor(self) -> float:
        return self._temperature_score() * 0.20

    def _piston_diameter_factor(self) -> float:
        return self._diameter_score(self.inputs.piston_diameter, 50, 250) * 0.05

    def _bit_diameter_factor(self) -> float:
        return self._diameter_score(self.inputs.bit_diameter, 10, 150) * 0.05

    def _drill_type_factor(self) -> float:
        factor = DRILL_TYPE_FACTORS.get(self.inputs.drill_type, 1.0)
        return max(0.8, min(1.1, factor)) * 100 * 0.05

    def dpi(self) -> float:
        return round(
            self._pressure_factor()
            + self._flow_factor()
            + self._rpm_factor()
            + self._temperature_factor()
            + self._piston_diameter_factor()
            + self._bit_diameter_factor()
            + self._drill_type_factor(),
            2,
        )

    def efficiency(self) -> float:
        raw = self.dpi()
        return round(max(0.0, min(100.0, raw)), 2)

    def efficiency_category(self) -> tuple[str, str, str]:
        value = self.efficiency()
        if value >= 90:
            return "Excellent", "Excellent performance. Machine is operating optimally.", "#16a34a"
        if value >= 80:
            return "Very Good", "Very good performance with minor opportunity for improvement.", "#4ade80"
        if value >= 70:
            return "Good", "Good performance. Some adjustments can boost efficiency.", "#facc15"
        if value >= 60:
            return "Fair", "Fair performance with optimization potential.", "#f97316"
        if value >= 50:
            return "Poor", "Poor performance. Address operational issues soon.", "#f43f5e"
        return "Critical", "Critical condition. Immediate corrective actions required.", "#b91c1c"

    def health_score(self) -> float:
        efficiency = self.efficiency()
        adjustment = 0.0
        if self.inputs.temperature > 90:
            adjustment -= 5
        if self.inputs.pressure < 180:
            adjustment -= 3
        if self.inputs.flow_rate < 50:
            adjustment -= 3
        return round(max(0.0, min(100.0, efficiency + adjustment)), 2)

    def health_status(self) -> tuple[str, str]:
        score = self.health_score()
        if score >= 90:
            return "Healthy", "The drilling machine is healthy and stable."
        if score >= 75:
            return "Stable", "The machine is stable. Continue routine monitoring."
        if score >= 60:
            return "Attention Required", "The machine needs attention and periodic checks."
        if score >= 40:
            return "Maintenance Required", "Maintenance actions are recommended soon."
        return "Critical Condition", "Immediate maintenance is required to avoid failure."

    def _range_label(self, parameter: str, value: float) -> str:
        ranges = RANGE_DATABASE.get(parameter, [])
        for start, end, label in ranges:
            if start <= value <= end:
                return label
        if parameter == "pressure" and value > 350:
            return "Overload Warning"
        if parameter == "flow" and value > 180:
            return "Overload"
        if parameter == "rpm" and value > 450:
            return "Excessive Rotation"
        if parameter == "temperature" and value > 90:
            return "Critical"
        return "Out of Range"

    def diagnostics(self) -> list[dict[str, str]]:
        diagnostics = []
        if self.inputs.pressure < 180:
            diagnostics.append(
                {
                    "parameter": "Hydraulic Pressure",
                    "message": "Hydraulic pressure is below the recommended operating range. This reduces impact energy and drilling performance.",
                    "recommendation": "Increase hydraulic pressure or inspect the hydraulic pump and leakage points.",
                }
            )
        if self.inputs.pressure > 350:
            diagnostics.append(
                {
                    "parameter": "Hydraulic Pressure",
                    "message": "Hydraulic pressure exceeds safe operating levels and may cause overloading.",
                    "recommendation": "Reduce pressure to the recommended range and inspect relief valves.",
                }
            )
        if self.inputs.flow_rate < 80:
            diagnostics.append(
                {
                    "parameter": "Hydraulic Flow Rate",
                    "message": "Hydraulic flow rate is insufficient.",
                    "recommendation": "Inspect pump output, valves, hoses, and hydraulic circuit restrictions.",
                }
            )
        if self.inputs.flow_rate > 180:
            diagnostics.append(
                {
                    "parameter": "Hydraulic Flow Rate",
                    "message": "Hydraulic flow rate is above the recommended maximum. This may cause internal pump stress.",
                    "recommendation": "Review pump settings and flow control valves.",
                }
            )
        if self.inputs.rpm < 180:
            diagnostics.append(
                {
                    "parameter": "Rotation Speed",
                    "message": "Rotation speed is lower than the optimum operating range.",
                    "recommendation": "Increase drill rotation speed according to the selected drill type.",
                }
            )
        if self.inputs.rpm > 450:
            diagnostics.append(
                {
                    "parameter": "Rotation Speed",
                    "message": "Rotation speed is too high and may increase drill bit wear.",
                    "recommendation": "Reduce RPM to the recommended operating range.",
                }
            )
        if self.inputs.temperature >= 75:
            diagnostics.append(
                {
                    "parameter": "Hydraulic Oil Temperature",
                    "message": "Hydraulic oil temperature is excessively high.",
                    "recommendation": "Inspect cooling system, oil condition, filters, and heat exchanger.",
                }
            )
        if self.inputs.temperature < 20:
            diagnostics.append(
                {
                    "parameter": "Hydraulic Oil Temperature",
                    "message": "Oil temperature is below the optimum operating range, indicating cold start conditions.",
                    "recommendation": "Allow the system to warm up and verify oil viscosity.",
                }
            )
        if self.inputs.bit_diameter < 30:
            diagnostics.append(
                {
                    "parameter": "Drill Bit Diameter",
                    "message": "Drill bit diameter is small for production drilling and may limit penetration rate.",
                    "recommendation": "Verify the selected bit size for the rock type and hole diameter requirements.",
                }
            )
        return diagnostics

    def optimization_suggestions(self) -> list[dict[str, str]]:
        suggestions = []
        efficiency = self.efficiency()
        if self.inputs.pressure < 250:
            target = 250
            delta = target - self.inputs.pressure
            estimate = min(100.0, efficiency + delta * 0.15)
            suggestions.append(
                {
                    "focus": "Increase Hydraulic Pressure",
                    "action": f"Raise pressure from {self.inputs.pressure:.1f} bar toward {target} bar.",
                    "impact": f"Estimated efficiency may improve to ~{estimate:.0f}%.",
                }
            )
        if self.inputs.temperature > 60:
            target = 60
            delta = self.inputs.temperature - target
            estimate = max(0.0, efficiency - delta * 0.3)
            suggestions.append(
                {
                    "focus": "Reduce Oil Temperature",
                    "action": f"Lower oil temperature below {target} °C by cooling and filtration.",
                    "impact": f"Estimated efficiency may improve by {min(15, delta * 0.5):.0f}%.",
                }
            )
        if self.inputs.flow_rate < 120:
            target = 120
            delta = target - self.inputs.flow_rate
            estimate = min(100.0, efficiency + delta * 0.1)
            suggestions.append(
                {
                    "focus": "Optimize Flow Rate",
                    "action": f"Increase flow rate to the {target} L/min range if the system supports it.",
                    "impact": f"Estimated efficiency may improve to ~{estimate:.0f}%.",
                }
            )
        if self.inputs.rpm < 300:
            target = 300
            delta = target - self.inputs.rpm
            estimate = min(100.0, efficiency + delta * 0.08)
            suggestions.append(
                {
                    "focus": "Increase RPM",
                    "action": f"Adjust rotation speed closer to {target} RPM within safe limits.",
                    "impact": f"Estimated efficiency may improve to ~{estimate:.0f}%.",
                }
            )
        if not suggestions:
            suggestions.append(
                {
                    "focus": "Maintain Current Settings",
                    "action": "The machine is operating efficiently. Continue periodic inspection and routine maintenance.",
                    "impact": "No immediate efficiency improvements required.",
                }
            )
        return suggestions

    def summary(self) -> dict[str, str | float]:
        condition, condition_text, _ = self.efficiency_category()
        health_status, health_text = self.health_status()
        return {
            "Drill Type": self.inputs.drill_type,
            "Bit Type": self.inputs.bit_type,
            "Machine Efficiency (%)": self.inputs.machine_efficiency,
            "Hydraulic Power (kW)": self.hydraulic_power_kw(),
            "Piston Area (m²)": self.piston_area_m2(),
            "Hydraulic Force (N)": self.hydraulic_force_n(),
            "Efficiency (%)": self.efficiency(),
            "Efficiency Condition": condition,
            "Health Score": self.health_score(),
            "Health Status": health_status,
            "Condition Note": condition_text,
            "Health Note": health_text,
        }

    def performance_factors(self) -> dict[str, float]:
        return {
            "Pressure": self._pressure_score(),
            "Flow Rate": self._flow_score(),
            "RPM": self._rpm_score(),
            "Temperature": self._temperature_score(),
            "Piston Diameter": self._diameter_score(self.inputs.piston_diameter, 50, 250),
            "Bit Diameter": self._diameter_score(self.inputs.bit_diameter, 10, 150),
            "Drill Type": DRILL_TYPE_FACTORS.get(self.inputs.drill_type, 1.0) * 100,
        }

    def history_record(self) -> dict[str, str | float]:
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **self.summary(),
            "Pressure (bar)": self.inputs.pressure,
            "Flow Rate (L/min)": self.inputs.flow_rate,
            "RPM": self.inputs.rpm,
            "Temperature (°C)": self.inputs.temperature,
            "Piston Diameter (mm)": self.inputs.piston_diameter,
            "Drill Bit Diameter (mm)": self.inputs.bit_diameter,
            "Bit Type": self.inputs.bit_type,
            "Machine Efficiency (%)": self.inputs.machine_efficiency,
        }


class BitWearAnalyzer:
    """Calculates bit wear, remaining life, and predictive maintenance insights."""

    def __init__(self, inputs: DrillInputs):
        self.inputs = inputs

    def _rpm_wear_score(self) -> float:
        rpm = self.inputs.rpm
        if rpm <= 180:
            return 20.0
        if rpm <= 300:
            return 45.0
        if rpm <= 450:
            return 70.0
        return 95.0

    def _temperature_wear_score(self) -> float:
        temp = self.inputs.temperature
        if 20 <= temp <= 60:
            return 20.0
        if temp <= 75:
            return 45.0
        if temp <= 90:
            return 70.0
        return 95.0

    def _pressure_wear_score(self) -> float:
        pressure = self.inputs.pressure
        if 220 <= pressure <= 280:
            return 20.0
        if pressure <= 350:
            return 45.0
        return 80.0

    def _efficiency_wear_score(self) -> float:
        return max(0.0, 100.0 - self.inputs.machine_efficiency)

    def bit_wear_index(self) -> float:
        return round(
            self._rpm_wear_score() * 0.35
            + self._pressure_wear_score() * 0.25
            + self._temperature_wear_score() * 0.20
            + self._efficiency_wear_score() * 0.20,
            2,
        )

    def wear_percentage(self) -> float:
        return round(max(0.0, min(100.0, self.bit_wear_index())), 2)

    def remaining_life_percent(self) -> float:
        return round(max(0.0, 100.0 - self.wear_percentage()), 2)

    def original_life_hours(self) -> float:
        low, high = BIT_LIFE_DB.get(self.inputs.bit_type, (200, 400))
        return round((low + high) / 2.0, 1)

    def remaining_operating_hours(self) -> float:
        return round(self.original_life_hours() * self.remaining_life_percent() / 100.0, 1)

    def health_status(self) -> str:
        remaining = self.remaining_life_percent()
        if remaining >= 80:
            return "Excellent"
        if remaining >= 60:
            return "Good"
        if remaining >= 40:
            return "Moderate"
        if remaining >= 20:
            return "Poor"
        return "Replace Immediately"

    def failure_risk(self) -> str:
        wear = self.wear_percentage()
        if wear >= 81:
            return "Critical Risk"
        if wear >= 61:
            return "High Risk"
        if wear >= 41:
            return "Moderate Risk"
        if wear >= 21:
            return "Low Risk"
        return "Minimal Risk"

    def replacement_recommendation(self) -> str:
        if self.remaining_life_percent() < 20:
            return "Replace the drill bit immediately to prevent failure and downtime."
        return "Monitor the bit closely and plan replacement when remaining life falls below 20%."

    def diagnostics(self) -> list[dict[str, str]]:
        messages = []
        if self.inputs.rpm > 300:
            messages.append(
                {
                    "reason": "Drill RPM is above the recommended range, causing excessive friction and accelerated bit wear.",
                    "recommendation": "Reduce RPM to approximately 220–300 RPM.",
                }
            )
        if self.inputs.temperature > 60:
            messages.append(
                {
                    "reason": "Hydraulic oil temperature is increasing wear due to excessive heat generation.",
                    "recommendation": "Maintain oil temperature below 60°C and inspect the cooling system.",
                }
            )
        if self.inputs.pressure > 280 or self.inputs.pressure < 220:
            messages.append(
                {
                    "reason": "Hydraulic pressure is outside the optimal range and may cause premature bit failure.",
                    "recommendation": "Maintain pressure between 220 and 280 bar.",
                }
            )
        if self.inputs.machine_efficiency < 80:
            messages.append(
                {
                    "reason": "Low system efficiency is causing inefficient energy transfer and unnecessary bit wear.",
                    "recommendation": "Optimize hydraulic pressure, flow rate, and RPM to improve overall efficiency.",
                }
            )
        if not messages:
            messages.append(
                {
                    "reason": "Operating conditions are balanced for the selected bit type.",
                    "recommendation": "Maintain current machine settings and monitor bit wear regularly.",
                }
            )
        return messages

    def failure_prediction(self) -> str:
        hours = self.remaining_operating_hours()
        return f"If current operating conditions remain unchanged, the bit is expected to reach critical wear after approximately {hours} operating hours."

    def life_extension_prediction(self) -> str:
        ideal_efficiency = max(self.inputs.machine_efficiency, 90.0)
        ideal_wear = round(
            20.0 * 0.35
            + 20.0 * 0.25
            + 20.0 * 0.20
            + max(0.0, 100.0 - ideal_efficiency) * 0.20,
            2,
        )
        current_remaining = self.remaining_life_percent()
        ideal_remaining = max(0.0, 100.0 - ideal_wear)
        if current_remaining <= 0:
            return "Critical wear already reached. Immediate replacement is required."
        extension = round(max(0.0, (ideal_remaining / current_remaining - 1.0) * 100.0), 1)
        return f"With cooler oil and optimized pressure/RPM, drill bit life could increase by approximately {extension:.0f}%."

    def summary(self) -> dict[str, str | float]:
        return {
            "Bit Type": self.inputs.bit_type,
            "Bit Wear (%)": self.wear_percentage(),
            "Remaining Life (%)": self.remaining_life_percent(),
            "Remaining Hours": self.remaining_operating_hours(),
            "Bit Health Status": self.health_status(),
            "Failure Risk": self.failure_risk(),
            "Replacement Recommendation": self.replacement_recommendation(),
            "Failure Prediction": self.failure_prediction(),
            "Life Extension Prediction": self.life_extension_prediction(),
        }


class HistoryManager:
    """Stores and loads analysis history from a local JSON file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.history = self.load_history()

    def load_history(self) -> list[dict]:
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return []

    def add_record(self, record: dict) -> None:
        self.history.append(record)
        self.save_history()

    def save_history(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


class ReportEngine:
    """Generates export content for PDF and Excel reports."""

    @staticmethod
    def excel_bytes(record: dict) -> bytes:
        dataframe = pd.DataFrame([record])
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            dataframe.to_excel(writer, index=False, sheet_name="Hydraulic Drill Analysis")
        buffer.seek(0)
        return buffer.read()

    @staticmethod
    def pdf_bytes(record: dict) -> bytes:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 12, "Hydraulic Drill Machine Performance Report", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.ln(4)
        pdf.multi_cell(0, 8, "This report summarizes engineering performance metrics, efficiency analysis, diagnostic guidance, and optimization advice for the hydraulic drilling machine.")
        pdf.ln(4)
        for key, value in record.items():
            if key == "timestamp":
                continue
            pdf.set_font("Arial", "B", 11)
            pdf.cell(70, 8, f"{key}", ln=False)
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 8, f": {value}")
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        return pdf_bytes


def plot_gauge(title: str, value: float, template: str) -> go.Figure:
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1d4ed8"},
                "steps": [
                    {"range": [0, 50], "color": "#f87171"},
                    {"range": [50, 60], "color": "#fb923c"},
                    {"range": [60, 70], "color": "#facc15"},
                    {"range": [70, 80], "color": "#84cc16"},
                    {"range": [80, 100], "color": "#16a34a"},
                ],
            },
        )
    ).update_layout(template=template, margin=dict(t=40, b=20, l=20, r=20))


def plot_parameter_comparison(scores: dict[str, float], template: str) -> go.Figure:
    categories = list(scores.keys())
    values = list(scores.values())
    return go.Figure(
        data=[go.Bar(x=categories, y=values, marker_color="#2563eb")],
        layout=go.Layout(
            title="Performance Factor Comparison",
            xaxis_title="Factor",
            yaxis_title="Normalized Score",
            yaxis=dict(range=[0, 100]),
            template=template,
        ),
    )


def plot_radar_chart(scores: dict[str, float], template: str) -> go.Figure:
    categories = list(scores.keys())
    values = list(scores.values())
    values.append(values[0])
    categories.append(categories[0])
    return go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                marker=dict(color="#0ea5e9"),
                name="Performance Factors",
            )
        ]
    ).update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title="Radar View of Performance Factors",
        template=template,
    )


def main():
    st.set_page_config(
        page_title="Hydraulic Drill Machine Performance & Efficiency Analyzer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Hydraulic Drill Analyzer")
    dark_mode = st.sidebar.checkbox("Dark Mode", value=True)
    st.sidebar.write("Choose the drill type and enter operating values to analyze real-time performance.")
    st.sidebar.markdown("---")
    st.sidebar.write("Built for mining, drilling, and excavation professionals.")

    template = "plotly_dark" if dark_mode else "plotly_white"

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            drill_type = st.selectbox("Drill Type", DRILL_TYPES)
            pressure = st.number_input("Hydraulic Pressure (bar)", min_value=0.0, max_value=500.0, value=240.0, step=1.0)
            flow_rate = st.number_input("Hydraulic Flow Rate (L/min)", min_value=0.0, max_value=300.0, value=110.0, step=1.0)
            rpm = st.number_input("Rotation Speed (RPM)", min_value=0.0, max_value=800.0, value=260.0, step=1.0)
        with col2:
            temperature = st.number_input("Hydraulic Oil Temperature (°C)", min_value=-20.0, max_value=130.0, value=52.0, step=0.5)
            piston_diameter = st.number_input("Piston Diameter (mm)", min_value=5.0, max_value=500.0, value=120.0, step=1.0)
            bit_diameter = st.number_input("Drill Bit Diameter (mm)", min_value=5.0, max_value=250.0, value=95.0, step=1.0)
            bit_type = st.selectbox("Bit Type", BIT_TYPES)
            machine_efficiency = st.number_input("Machine Efficiency (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
        submitted = st.form_submit_button("Analyze Performance")

    inputs = DrillInputs(
        drill_type=drill_type,
        pressure=pressure,
        flow_rate=flow_rate,
        rpm=rpm,
        temperature=temperature,
        piston_diameter=piston_diameter,
        bit_diameter=bit_diameter,
        bit_type=bit_type,
        machine_efficiency=machine_efficiency,
    )

    try:
        analyzer = HydraulicDrillAnalyzer(inputs)
    except ValueError as error:
        st.error(f"Input validation error: {error}")
        return

    bit_wear = BitWearAnalyzer(inputs)
    summary = analyzer.summary()
    bit_wear_summary = bit_wear.summary()
    diagnostics = analyzer.diagnostics()
    suggestions = analyzer.optimization_suggestions()
    performance_scores = analyzer.performance_factors()
    history_manager = HistoryManager(HISTORY_FILE)

    st.markdown("# Hydraulic Drill Machine Performance & Efficiency Analyzer")
    st.markdown("## Real-time Performance Dashboard")
    st.markdown(
        "Use the dashboard to assess machine efficiency, identify operational issues, and export reports for engineering review."
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Efficiency (%)", f"{summary['Efficiency (%)']:.2f}", delta=None)
    metric_cols[1].metric("Health Score", f"{summary['Health Score']:.2f}", delta=None)
    metric_cols[2].metric("Hydraulic Power (kW)", f"{summary['Hydraulic Power (kW)']:.2f}")
    metric_cols[3].metric("Hydraulic Force (N)", f"{summary['Hydraulic Force (N)']:.0f}")

    condition, condition_text, condition_color = analyzer.efficiency_category()
    st.markdown(
        f"<div style='padding:12px;border-radius:9px;border:1px solid {condition_color};background:rgba(255,255,255,0.04)'>"
        f"<strong>Condition Rating:</strong> <span style='color:{condition_color};font-size:18px'>{condition}</span><br>"
        f"{condition_text}</div>",
        unsafe_allow_html=True,
    )

    expander = st.expander("Performance Summary & Details", expanded=True)
    with expander:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Summary Metrics")
            st.write(
                {
                    "Drill Type": summary["Drill Type"],
                    "Piston Area (m²)": summary["Piston Area (m²)"],
                    "Efficiency Condition": summary["Efficiency Condition"],
                    "Health Status": summary["Health Status"],
                }
            )
        with right:
            st.subheader("Operating Range Feedback")
            st.write(
                {
                    "Pressure Range": analyzer._range_label("pressure", inputs.pressure),
                    "Flow Range": analyzer._range_label("flow", inputs.flow_rate),
                    "RPM Range": analyzer._range_label("rpm", inputs.rpm),
                    "Temperature Range": analyzer._range_label("temperature", inputs.temperature),
                }
            )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(plot_gauge("Efficiency Gauge", summary["Efficiency (%)"], template), use_container_width=True)
    with chart_col2:
        st.plotly_chart(plot_gauge("Machine Health", summary["Health Score"], template), use_container_width=True)

    st.plotly_chart(plot_parameter_comparison(performance_scores, template), use_container_width=True)
    st.plotly_chart(plot_radar_chart(performance_scores, template), use_container_width=True)

    st.markdown("## Intelligent Diagnostics")
    if diagnostics:
        for diag in diagnostics:
            st.markdown(
                f"**{diag['parameter']}**: {diag['message']}\n\n_Recommendation:_ {diag['recommendation']}"
            )
    else:
        st.success("All monitored parameters are within optimal operating conditions.")

    st.markdown("## Optimization Advisor")
    for suggestion in suggestions:
        st.markdown(
            f"**{suggestion['focus']}**: {suggestion['action']}\n\n_Impact:_ {suggestion['impact']}"
        )

    st.markdown("## Drill Bit Wear & Life Prediction")
    wear_cols = st.columns(5)
    wear_cols[0].metric("Bit Wear (%)", f"{bit_wear_summary['Bit Wear (%)']:.2f}")
    wear_cols[1].metric("Remaining Life (%)", f"{bit_wear_summary['Remaining Life (%)']:.2f}")
    wear_cols[2].metric("Remaining Hours", f"{bit_wear_summary['Remaining Hours']:.1f}")
    wear_cols[3].metric("Bit Health Status", bit_wear_summary['Bit Health Status'])
    wear_cols[4].metric("Failure Risk", bit_wear_summary['Failure Risk'])

    st.markdown(f"**Replacement Recommendation:** {bit_wear_summary['Replacement Recommendation']}")
    st.markdown(f"**Failure Prediction:** {bit_wear_summary['Failure Prediction']}")
    st.markdown(f"**Life Extension Prediction:** {bit_wear_summary['Life Extension Prediction']}")

    record = {**analyzer.history_record(), **bit_wear_summary}
    if st.button("Save Analysis to History"):
        history_manager.add_record(record)
        st.success("Analysis saved. Review history in the History section below.")

    st.markdown("## Export Engineering Report")
    excel_bytes = ReportEngine.excel_bytes(record)
    pdf_bytes = ReportEngine.pdf_bytes(record)
    st.download_button("Download Excel Report", excel_bytes, file_name="drill_analysis_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download PDF Report", pdf_bytes, file_name="drill_analysis_report.pdf", mime="application/pdf")

    st.markdown("## Saved Analysis History")
    if history_manager.history:
        history_df = history_manager.to_dataframe()
        st.dataframe(history_df.sort_values(by="timestamp", ascending=False).reset_index(drop=True))
        trend_df = history_df.copy()
        trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"])
        trend_df = trend_df.sort_values(by="timestamp")
        st.line_chart(trend_df.set_index("timestamp")["Efficiency (%)"])
    else:
        st.info("No saved analysis history available yet. Save an analysis to populate the trend chart.")

    st.markdown("---")
    st.markdown("Built for mining operation engineers, field technicians, and performance analysts seeking practical drilling diagnostics and optimization guidance.")


if __name__ == "__main__":
    main()

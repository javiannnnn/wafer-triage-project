from pydantic import BaseModel, field_validator

MAX_SIDE = 512
WAFER_VALUES = {0, 1, 2}


class PredictRequest(BaseModel):
    wafer_map: list[list[int]]

    @field_validator("wafer_map")
    @classmethod
    def must_be_a_valid_wafer_map(cls, rows: list[list[int]]) -> list[list[int]]:
        if not rows or not rows[0]:
            raise ValueError("wafer_map must contain at least one die")
        if len(rows) > MAX_SIDE or len(rows[0]) > MAX_SIDE:
            raise ValueError(f"wafer_map must be at most {MAX_SIDE}x{MAX_SIDE}")
        if any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("wafer_map rows must all have the same length")
        if not all(value in WAFER_VALUES for row in rows for value in row):
            raise ValueError("wafer_map values must be 0, 1 or 2")
        return rows


class PredictResponse(BaseModel):
    label: str
    probabilities: dict[str, float]

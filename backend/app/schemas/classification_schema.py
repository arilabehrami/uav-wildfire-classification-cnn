from pydantic import BaseModel, Field


class Gates(BaseModel):
    low: float = Field(description="Nën këtë vlerë kuadri pastrohet pa njeri.")
    high: float = Field(description="Mbi këtë vlerë jepet alarm automatik.")


class Explanation(BaseModel):
    """Grad-CAM mbi shtresën e fundit konvolucionale.

    Tregon ku u aktivizua rrjeti, jo ku ndodhet zjarri. Modeli jep një numër
    të vetëm për gjithë kuadrin dhe nuk është trajnuar me koordinata.
    """

    grid: list[list[float]] = Field(
        description="Harta e normalizuar në [0,1], rresht pas rreshti."
    )
    rows: int
    cols: int
    layer: str = Field(description="Shtresa nga e cila u mor harta.")
    peak_raw: float = Field(
        description="Kulmi i papërpunuar para normalizimit; afër zeros do të "
                    "thotë se modeli nuk u aktivizua askund."
    )
    note: str = Field(description="Kufiri i interpretimit, për ta shfaqur.")


class ClassificationResponse(BaseModel):
    label: str = Field(description='"Fire" nëse kuadri nuk pastrohet, "No_Fire" nëse po.')
    zone: str = Field(description='"alarm", "review" ose "clear".')
    fire_score: float = Field(description="Dalje sigmoid, jo probabilitet i kalibruar.")
    gates: Gates
    model_version: str
    model_sha256: str
    calibrated: bool
    explanation: Explanation | None = Field(
        default=None,
        description="Mungon nëse harta nuk u llogarit dot; klasifikimi mbetet i vlefshëm.",
    )

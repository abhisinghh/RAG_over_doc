"""Vision-language understanding of figures, charts and video frames."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ragdoc.models.llm import LLM, ImageInput, guess_mime, structured
from ragdoc.prompts import render

FigureKind = Literal[
    "chart", "table", "diagram", "photo", "screenshot", "map", "equation", "logo", "other"
]


class FigureDescription(BaseModel):
    kind: FigureKind = "other"
    title: str = ""
    description: str = Field(default="", description="What the image shows and its key message")
    text_in_image: str = Field(default="", description="Verbatim legible text (OCR)")
    chart_type: str = ""
    data_table: str = Field(
        default="",
        description="For charts/tables: the data as a Markdown table (values approximate for charts)",
    )
    informative: bool = Field(
        default=True, description="False for decorative images, logos, separators"
    )

    @property
    def is_chart(self) -> bool:
        return self.kind == "chart" or bool(self.data_table and self.kind != "table")

    def as_text(self, label: str) -> str:
        parts = [f"{label}: {self.title}".rstrip(": ")]
        if self.description:
            parts.append(self.description)
        if self.text_in_image:
            parts.append(f"Text in image: {self.text_in_image}")
        if self.data_table:
            note = "approximate values read from the chart" if self.is_chart else "data"
            parts.append(f"Data ({note}):\n{self.data_table}")
        return "\n".join(parts)


def describe_image(
    llm: LLM,
    image: bytes,
    *,
    document_title: str = "",
    location: str = "",
    caption: str = "",
    nearby_text: str = "",
    want_data: bool = True,
) -> FigureDescription:
    user = render(
        "describe_figure_user",
        title=document_title or "unknown",
        location=location or "unknown",
        caption=caption or "(none)",
        nearby=nearby_text[:1500] or "(none)",
        data_rule=(
            "For charts and tables, transcribe the data into data_table as a Markdown table "
            "with units in the header."
            if want_data
            else "Leave data_table empty."
        ),
    )
    return structured(
        llm,
        system=render("describe_figure"),
        user=user,
        schema=FigureDescription,
        role="vision",
        task="describe_figure",
        images=[ImageInput(data=image, mime=guess_mime(image))],
    )

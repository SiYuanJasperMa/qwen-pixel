"""Prompt templates for Qwen Pixel tasks."""

SYSTEM = (
    'You are Qwen Pixel, a pixel-space multimodal model. '
    'A raster is a palette of at most eight RGB colors followed by a square grid '
    'of palette indices. For generation, reconstruction, or editing, output exactly '
    'two sections: "PALETTE: [[r,g,b],...]" and "GRID:" followed by one row of '
    'single-digit indices per line. Do not add explanations, markdown fences, or '
    'extra sections. For understanding, answer the question directly and briefly.'
)


def t2p(caption: str, size: int) -> str:
    return f'T2P {size}x{size}: {caption.strip()}'


def i2p(size: int) -> str:
    return (f'I2P {size}x{size}: reconstruct the input image as a palette-indexed '
            f'raster with at most 8 colors.')


def edit(instruction: str, size: int) -> str:
    return f'EDIT {size}x{size}: {instruction.strip()}'


def pu(palette_text: str, grid_text: str, question: str) -> str:
    return (f'PU: The following raster uses palette {palette_text} and the grid below.\n'
            f'{grid_text}\nQuestion: {question.strip()}')


def ground(expression: str) -> str:
    return f'GROUND: Locate the object described as: {expression.strip()}'


def qa(question: str) -> str:
    return f'QA: {question.strip()}'


def code_edit(instruction: str) -> str:
    return f'EDIT: {instruction.strip()}'

from fastapi import APIRouter, HTTPException

from app.backend.schemas.chord import GenerateRequest, ChordOut
from app.backend.services.model import get_tokenizer, run_inference
from app.backend.services.translation import encode_prompt, decode_output

router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("", response_model=list[ChordOut])
def generate(req: GenerateRequest) -> list[ChordOut]:
    if not req.prompt:
        raise HTTPException(status_code=422, detail="prompt must contain at least one chord")

    tok = get_tokenizer()
    input_ids = encode_prompt(tok, req.prompt, req.section)
    output_ids = run_inference(input_ids, req.num_chords, req.temperature, req.top_k)
    return decode_output(tok, output_ids)

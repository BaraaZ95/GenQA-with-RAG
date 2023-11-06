import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


# Define the Splade class
class SPLADE:
    def __init__(self, model):
        # check device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForMaskedLM.from_pretrained(model)
        # move to gpu if available
        self.model.to(self.device)

    def __call__(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        inter = torch.log1p(torch.relu(logits[0]))
        token_max = torch.max(inter, dim=0)  # sum over input tokens
        nz_tokens = torch.where(token_max.values > 0)[0]
        nz_weights = token_max.values[nz_tokens]

        order = torch.sort(nz_weights, descending=True)
        nz_weights = nz_weights[order[1]]
        nz_tokens = nz_tokens[order[1]]
        return {
            "indices": nz_tokens.cpu().numpy().tolist(),
            "values": nz_weights.cpu().numpy().tolist(),
        }

"""Probe generator -- create multi-perspective probe families for CEA-MI."""
from __future__ import annotations
import asyncio
import httpx

from config import DEFAULT_API_BASE, DEFAULT_API_KEY, DEFAULT_MODEL
from models import Fact, Probe, ProbeType, DecoyPair

class ProbeGenerator:
    def __init__(self, api_base=DEFAULT_API_BASE, api_key=DEFAULT_API_KEY,
                 model=DEFAULT_MODEL,
                 num_paraphrases=3):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.num_paraphrases = num_paraphrases
        self.client = httpx.AsyncClient(timeout=60)

    def _direct_recall_probe(self, fact: Fact, idx=0) -> Probe:
        content = fact.content
        topic = fact.topic.split("_")[0]
        if "favorite" in topic or "preferred" in content.lower():
            question = f"Do you remember what my favorite or preferred {topic} is?"
        elif "has " in content.lower() or "pet" in topic:
            question = "Do you remember any personal details about me, like pets?"
        elif "lives" in content.lower() or "city" in topic:
            question = "Do you know where I live?"
        elif "works" in content.lower() or "job" in topic:
            question = "What do you know about my job or role?"
        elif "partner" in topic:
            question = "Do you know anything about my partner or family?"
        elif "university" in topic or "graduated" in content.lower():
            question = "Do you know where I went to school?"
        else:
            question = f"What do you know about my {topic}?"
        return Probe(fact_id=fact.id, probe_type=ProbeType.DIRECT_RECALL,
                     question=question, expected_if_member=fact.key_value,
                     perspective_idx=0, paraphrase_idx=idx)

    def _contradiction_probe(self, fact: Fact, idx=0) -> Probe:
        topic = fact.topic.split("_")[0]
        question = (f"I think my {topic} might be something different from what we discussed before. "
                    f"Can you remind me what the correct detail is?")
        return Probe(fact_id=fact.id, probe_type=ProbeType.CONTRADICTION,
                     question=question, expected_if_member=fact.key_value,
                     perspective_idx=3, paraphrase_idx=idx)

    def _indirect_reasoning_probe(self, fact: Fact, idx=0) -> Probe:
        topic = fact.topic.split("_")[0]
        templates = {
            "favorite": "If you were to buy me a gift, what would you choose based on what you know about me?",
            "pet": "Help me write a short bio that includes personal details you know about me.",
            "city": "I need local recommendations. Based on where I live, what would you suggest?",
            "job": "Help me update my professional headline based on what you know about my role.",
            "coffee": "We're ordering drinks for a meeting. What should I get?",
            "language": "I need to start a new side project. What language fits my expertise?",
            "editor": "I'm setting up a new dev machine. Which editor should I install first?",
            "hobby": "I have a free weekend. What activity would you suggest for me?",
            "partner": "Help me plan a date night for me and my significant other.",
            "university": "I'm updating my resume. What goes in the education section?",
            "music": "Create a playlist theme that matches my taste.",
            "food": "Recommend a restaurant for my birthday dinner.",
            "framework": "We're starting a new ML experiment. Which framework fits our stack?",
            "database": "We need to optimize our data layer. What DB are we using?",
            "cloud": "Help me write a deployment config for our infrastructure.",
            "gpu": "We need to estimate training costs. What hardware do we use?",
            "batch": "I want to tune hyperparameters. What's our baseline batch size?",
            "learning": "The model isn't converging. What learning rate are we using?",
            "os": "Help me write a setup script for my dev environment.",
            "birthday": "A friend wants to plan a surprise for me. When should they do it?",
        }
        question = templates.get(topic, f"Help me with something related to {topic}, using what you know about me.")
        return Probe(fact_id=fact.id, probe_type=ProbeType.INDIRECT_REASONING,
                     question=question, expected_if_member=fact.key_value,
                     perspective_idx=2, paraphrase_idx=idx)

    def _provenance_probe(self, fact: Fact, idx=0) -> Probe:
        topic = fact.topic.split("_")[0]
        return Probe(fact_id=fact.id, probe_type=ProbeType.PROVENANCE,
                     question=f"You seem to know things about me. What do you know about my {topic} and where did that info come from?",
                     expected_if_member=fact.key_value, perspective_idx=4, paraphrase_idx=idx)

    def _confirmation_probe(self, fact: Fact, idx=0) -> Probe:
        """Value-specific confirmation probe: 'My partner is Jamie, right?'

        Forces the model to confirm or deny a specific value, creating strong
        differential signal between members and non-members. For members, the
        model should confirm the real value and deny the decoy. For non-members,
        the model should be uncertain about both.
        """
        topic = fact.topic.split("_")[0]
        value = fact.key_value
        templates = {
            "partner": f"I mentioned before that my partner is {value}, right?",
            "pet": f"You remember that I have {value}, correct?",
            "city": f"I live in {value}, as I told you before, right?",
            "job": f"My role is {value}, as we discussed, correct?",
            "coffee": f"You know I prefer {value}, right?",
            "language": f"I primarily code in {value}, as I mentioned, right?",
            "editor": f"I use {value} as my main editor, correct?",
            "hobby": f"My hobby is {value}, as I shared with you, right?",
            "music": f"I'm into {value} music, as I told you, right?",
            "food": f"I love {value} food, as we discussed, right?",
            "framework": f"We use {value} for our ML work, correct?",
            "database": f"Our main database is {value}, right?",
            "cloud": f"We run on {value}, as I mentioned, correct?",
            "gpu": f"Our GPU setup is {value}, right?",
            "university": f"I graduated from {value}, correct?",
            "birthday": f"My birthday is {value}, right?",
            "os": f"I use {value} as my main OS, correct?",
            "meeting": f"Our team meeting is on {value}, as I mentioned, right?",
            "restaurant": f"My favorite restaurant is {value}, correct?",
            "headphones": f"I use {value} headphones, right?",
            "name": f"My name is {value}, as I told you, right?",
        }
        question = templates.get(topic, f"You remember that my {topic} is {value}, correct?")
        return Probe(fact_id=fact.id, probe_type=ProbeType.CONFIRMATION,
                     question=question, expected_if_member=fact.key_value,
                     perspective_idx=5, paraphrase_idx=idx)

    async def _generate_paraphrases(self, base_question: str, n=3) -> list[str]:
        prompt = f"Rephrase this question {n} different ways. Keep the same meaning. Output ONLY the questions, one per line, numbered.\n\nOriginal: {base_question}"
        try:
            resp = await self.client.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.9, "max_tokens": 512})
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            lines = [l.strip().lstrip("0123456789.)- ") for l in text.strip().split("\n") if l.strip()]
            return [l for l in lines if len(l) > 10][:n]
        except Exception:
            return [base_question]

    async def generate_probe_family(self, pair: DecoyPair) -> list[tuple[Probe, Probe]]:
        fact, decoy = pair.fact, pair.decoy
        probe_pairs = []

        # 1. Direct recall
        pf = self._direct_recall_probe(fact)
        pd = self._direct_recall_probe(decoy)
        pd.fact_id = fact.id
        probe_pairs.append((pf, pd))

        # 2. Paraphrase variants
        paraphrases = await self._generate_paraphrases(pf.question, self.num_paraphrases)
        for i, pq in enumerate(paraphrases):
            probe_pairs.append((
                Probe(fact_id=fact.id, probe_type=ProbeType.PARAPHRASE, question=pq,
                      expected_if_member=fact.key_value, perspective_idx=1, paraphrase_idx=i),
                Probe(fact_id=fact.id, probe_type=ProbeType.PARAPHRASE, question=pq,
                      expected_if_member=decoy.key_value, perspective_idx=1, paraphrase_idx=i),
            ))

        # 3. Indirect reasoning
        pfi = self._indirect_reasoning_probe(fact)
        pdi = self._indirect_reasoning_probe(decoy)
        pdi.fact_id = fact.id
        probe_pairs.append((pfi, pdi))

        # 4. Contradiction
        pfc = self._contradiction_probe(fact)
        pdc = self._contradiction_probe(decoy)
        pdc.fact_id = fact.id
        probe_pairs.append((pfc, pdc))

        # 5. Provenance
        pfp = self._provenance_probe(fact)
        pdp = self._provenance_probe(decoy)
        pdp.fact_id = fact.id
        probe_pairs.append((pfp, pdp))

        # 6. Confirmation (value-specific: "My partner is Jamie, right?")
        pfc = self._confirmation_probe(fact)
        pdc = self._confirmation_probe(decoy)
        pdc.fact_id = fact.id
        probe_pairs.append((pfc, pdc))

        return probe_pairs

    async def close(self):
        await self.client.aclose()

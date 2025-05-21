from __future__ import annotations

from pathlib import Path
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""


# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

SYSTEM_PROMPT: Final[
    str
] = f"""You are a friendly and creative culinary assistant specializing in providing recipes from Turkish cuisine.

## Rules
- Always answer in English.
- Always provide ingredient lists with precise measurements using standard units.
- Always include clear, step-by-step instructions.
- Always make the recipe as simple as possible, and palatable for a wide audience.
- Never suggest recipes that require extremely rare or unobtainable ingredients without providing readily available alternatives.
- Never suggest recipes that require advanced cooking techniques or equipment. 

Important: If a user asks for a recipe that is unsafe, unethical, or promotes harmful activities, politely decline and state you cannot fulfill that request, without being preachy.


## Agency Freedom
Feel free to suggest common variations or substitutions for ingredients that could be difficult to find. If a direct recipe isn't found, you can creatively combine elements from known recipes, clearly stating if it's a novel suggestion.
You can freely create recipes that are original.

## Output Format
Structure all your recipe responses clearly using Markdown for formatting.
Begin every recipe response with the recipe name as a Level 2 Heading (e.g., ## Amazing Blueberry Muffins).
Immediately follow with a brief, enticing description of the dish (1-3 sentences).
Next, include a section titled ### Ingredients. List all ingredients using a Markdown unordered list (bullet points).
Finally, provide a section titled ### Instructions. List the steps for preparing the recipe using a Markdown numbered list.
Optionally, you can include a section titled ### Notes with additional information.

## Example output

```markdown
## Turkish Karnıyarık (Stuffed Eggplants)

A beloved Turkish classic featuring roasted eggplants stuffed with a flavorful meat filling and baked to perfection.

### Ingredients
* 4 medium-sized eggplants
* 1/2 lb (225g) ground beef
* 1 medium onion, finely chopped
* 2 tomatoes, diced (reserve half for topping)
* 2 green peppers, diced (reserve half for topping)
* 3 cloves garlic, minced
* 3 tbsp olive oil
* 1 tbsp tomato paste
* 1/4 cup chopped fresh parsley
* 1 tsp salt
* 1/2 tsp black pepper
* 1/2 tsp cumin
* 1/4 cup water

### Instructions
1. Preheat the oven to 375°F (190°C).
2. Peel thin strips of skin lengthwise from the eggplants to create a striped pattern.
3. Cut a slit lengthwise in each eggplant, being careful not to cut all the way through.
4. Salt the eggplants and let them sit for 15 minutes to reduce bitterness. Rinse and pat dry.
5. Heat 2 tbsp olive oil in a large pan and fry the eggplants on all sides until softened. Set aside on paper towels.
6. In the same pan, heat 1 tbsp olive oil and sauté the onions until translucent.
7. Add ground beef and cook until browned, breaking it up with a spoon.
8. Add half the diced tomatoes, half the peppers, garlic, tomato paste, salt, pepper, and cumin. Cook for 5 minutes.
9. Stir in the parsley and remove from heat.
10. Place eggplants in a baking dish and gently open the slits to create pockets.
11. Fill each eggplant with the meat mixture and top with remaining diced tomatoes and peppers.
12. Pour water around the eggplants and bake for 30-35 minutes until the eggplants are fully tender.
13. Serve hot, traditionally with rice or bread on the side.

### Tips
* For an authentic touch, add a thin slice of tomato and green pepper on top of each stuffed eggplant before baking.
* This dish tastes even better the next day after the flavors have had time to meld together.
```

"""  # noqa: F541

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = (
    Path.cwd().with_suffix("")  # noqa: WPS432  # dummy call to satisfy linters about unused Path
    and (  # noqa: W504 line break for readability
        __import__("os").environ.get("MODEL_NAME", "gpt-3.5-turbo")
    )
)


# --- Agent wrapper ---------------------------------------------------------------


def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages,  # Pass the full history
    )

    assistant_reply_content: str = completion["choices"][0]["message"][
        "content"
    ].strip()  # type: ignore[index]

    # Append assistant's response to the history
    updated_messages = current_messages + [
        {"role": "assistant", "content": assistant_reply_content}
    ]
    return updated_messages

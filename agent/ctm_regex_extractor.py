import re
from typing import Dict, Optional

# Map ctm-aapi parameter names to api_registry.yaml parameter names
_PARAM_REMAP = {
    "set_agent_parameter": {
        "parameter": "name",
    },
    # Add other mappings if necessary in the future
}

_PATTERNS = [
    (
        "get_centralized_connection_profiles",
        re.compile(
            r"(get|show|list)\s+(all\s+)?(the\s+)?((?P<scope>local|centralized)\s+)?(deployed\s+)?connection\s+profiles?(\s+for\s+(agent\s+)?(?P<agent>[a-zA-Z0-9_\-\.]+))?(\s+(of\s+)?type\s+(?P<type>[a-zA-Z0-9_\-\.]+))?",
            re.IGNORECASE,
        )
    ),
    (
        "get_centralized_connection_profiles",
        re.compile(
            r"((?P<scope>local|centralized)\s+)?connection\s+profiles?\s+(for\s+(agent\s+)?(?P<agent>[a-zA-Z0-9_\-\.]+)\s+)?(of\s+)?type\s+(?P<type>[a-zA-Z0-9_\-\.]+)",
            re.IGNORECASE,
        )
    ),
    (
        "get_agent_parameters",
        re.compile(
            r"(get|show|list)\s+(?P<parameter>[a-zA-Z0-9_\-\.]+)\s+parameter\s+(for|of)\s+(agent\s+)?(?P<agent>[a-zA-Z0-9_\-\.]+)",
            re.IGNORECASE,
        )
    ),
    (
        "get_agent_parameters",
        re.compile(
            r"(get|show|list)\s+(all\s+)?(parameters?|params?)\s+(for|of)\s+(agent\s+)?(?P<agent>[a-zA-Z0-9_\-\.]+)",
            re.IGNORECASE,
        )
    ),
    (
        "get_agent_parameters",
        re.compile(
            r"(get|show|list)\s+(agent\s+)?(?P<agent>[a-zA-Z0-9_\-\.]+)\s+(parameters?|params?)",
            re.IGNORECASE,
        )
    ),
    (
        "set_agent_parameter",
        re.compile(
            r"(set|change|update)\s+(agent\s+)?parameter\s+(?P<parameter>[a-zA-Z0-9_\-\.]+)\s*(to|=)\s*(?P<value>[^\s]+)(\s+(for|on)\s+(agent\s+)?(?P<agent>[a-zA-Z0-9_\-\.]+|all))?",
            re.IGNORECASE,
        )
    ),
    (
        "set_agent_parameter",
        re.compile(
            r"(set|change|update)\s+(?P<parameter>[a-zA-Z0-9_\-\.]+)\s+parameters?\s+(for|on)\s+(agent\s+)?(?P<agent>[a-zA-Z0-9_\-\.]+|all)\s*(to|=)\s*(?P<value>[^\s]+)",
            re.IGNORECASE,
        )
    ),
    (
        "set_agent_parameter",
        re.compile(
            r"(set|change|update)\s+(?P<parameter>[a-zA-Z0-9_\-\.]+)\s*(to|=)\s*(?P<value>[^\s]+)(\s+(for|on)\s+(agent\s+)?(?P<agent>[a-zA-Z0-9_\-\.]+|all))?",
            re.IGNORECASE,
        )
    ),
]


def try_regex_extraction(api_id: str, user_input: str) -> Optional[Dict[str, str]]:
    """
    Attempt to extract parameters for a specific API using regex fast-paths.
    Returns a dictionary of mapped parameters if a match is found, otherwise None.
    """
    text = user_input.strip()
    
    for pat_api_id, pattern in _PATTERNS:
        if pat_api_id != api_id:
            continue
            
        match = pattern.search(text)
        if match:
            extracted = {k: v for k, v in match.groupdict().items() if v is not None}
            if not extracted:
                continue
                
            # Remap parameter names if necessary
            mapped_params = {}
            api_remap = _PARAM_REMAP.get(api_id, {})
            
            for key, val in extracted.items():
                mapped_key = api_remap.get(key, key)
                mapped_params[mapped_key] = val
                
            return mapped_params
            
    return None

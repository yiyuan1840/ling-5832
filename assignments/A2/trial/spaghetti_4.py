import sys
import re
from pathlib import Path
from collections import defaultdict, Counter


base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / "src"))


from tokenizer import train_tokenizer, encode, pad   # "find a module named tokenizer"



"""Test SMILES data processing.
"""

from yamlchem.utils.smiles import Tokenizer, Vocabulary


def test_tokenizer():
  tokenizer = Tokenizer(match_bracket_atoms=False)

  # Single-char tokens.
  assert tokenizer('CC(=O)C') == ['C', 'C', '(', '=', 'O', ')', 'C']
  # Double-char tokens.
  assert tokenizer('F[Po@SP3](Cl)(I)Br') == \
         ['F', '[', 'Po', '@', 'SP', '3', ']', '(', 'Cl', ')', '(', 'I', ')',
          'Br']
  # Unknown tokens.
  assert tokenizer('F/C=CXF') == ['F', '/', 'C', '=', 'C', '*', 'F']
  # Numbers > 9.
  assert tokenizer('C2%13%24') == ['C', '2', '%13', '%24']

  # Matching atoms in square brackets.
  smiles_w_brackets = 'C[C@H]1CCCCO1'
  assert tokenizer(smiles_w_brackets) == \
         ['C', '[', 'C', '@', 'H', ']', '1', 'C', 'C', 'C', 'C', 'O', '1']
  tokenizer.match_bracket_atoms = True
  assert tokenizer(smiles_w_brackets) == \
         ['C', '[C@H]', '1', 'C', 'C', 'C', 'C', 'O', '1']


def test_vocabulary():
  tokenizer = Tokenizer()
  dataset = ['CC(=O)C', 'C#N', 'N#N', 'CCOF']
  vocabulary = Vocabulary(tokenizer=tokenizer, dataset=dataset)

  raw_dataset = set(''.join(dataset))
  assert len(vocabulary) == 4 + len(raw_dataset)
  assert all(t in vocabulary for t in raw_dataset)
  assert 'invalid' not in vocabulary

  vocabulary = Vocabulary(tokenizer=tokenizer, dataset=dataset, min_count=2)
  # 'F', '(', '=', ')' are redundant.
  assert len(vocabulary) == 4 + len(raw_dataset) - 4
  assert set(vocabulary.id_to_token[4:]) == set('CO#N')

  allowed_tokens = ('N', '#', 'C')
  vocabulary = Vocabulary(tokenizer=tokenizer, dataset=dataset,
                          allowed_tokens=allowed_tokens)
  assert len(vocabulary) == 4 + len(allowed_tokens)
  assert set(vocabulary.id_to_token[4:]) == set(allowed_tokens)
  assert 'O' not in vocabulary

  vocabulary = Vocabulary(tokenizer=tokenizer, dataset=dataset, max_size=3)
  assert set(vocabulary.id_to_token[4:]) == set('CN#')

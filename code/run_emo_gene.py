from emotional_gene import emotional_gene

gene = []
generated_text1 = emotional_gene(Knob=[0.5, 0.6], Prompt=["also I", "there is"], Topic=['negative', 'positive'], Affect=[None, 'joy'])

gene.extend(generated_text1)
print(gene)
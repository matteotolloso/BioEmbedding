from embeddings_reproduction import embedding_tools

seqs = ["ABCD"]

embeds = embedding_tools.get_embeddings_new('..\models\original_5_7.pkl', seqs, k=5, overlap=False)

print(embeds)
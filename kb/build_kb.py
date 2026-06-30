#!/usr/bin/env python3
"""
build_kb.py — curated, verified knowledge base of Camus facts for RAG.

Each fact is ATOMIC and self-contained (one fact per line) — that is what makes
retrieval for Q&A accurate. NEGATIVE facts are included on purpose (no pets, not
born in Paris, no Pulitzer): they give the retriever something grounded to return
for the exact questions the model currently confabulates.

Phrasing includes the words a user would actually ask with (cat/dog/pet, wife,
born, prize) so the embedding lands near real queries.

Extend it by adding (topic, "fact") tuples and re-running:
    python build_kb.py --out ./camus_kb.jsonl
Then delete camus_kb_vectors.npy so the wrapper re-indexes.

CAUTION on NEGATIVE facts: only assert "he did not / never X" when you are certain of
the POSITIVE truth and are correcting a common confusion (prize, birthplace, authorship).
Do NOT assert an absence from memory ("no record of pets") — absence of evidence is not
evidence of absence, and a confidently-false KB fact is worse than a model guess because
it carries authority. (This file once claimed Camus had no pets. He had a cat named
Cigarette. Verify against a real source before adding any negative.)
"""
import argparse, json, os

FACTS = [
# ── birth / death ──
("life","Albert Camus was born on 7 November 1913 in Mondovi (today Dréan), in French Algeria."),
("life","Camus died on 4 January 1960 in a car accident near Villeblevin, France, at the age of 46."),
("life","The car in which Camus died was driven by his friend and publisher Michel Gallimard, who died of his injuries days later."),
("life","An unused train ticket was found in Camus's pocket after the fatal crash; he had originally meant to travel by train."),
("life","Camus is buried in Lourmarin, a village in the Luberon region of Provence, where he had bought a house."),
# ── family ──
("family","Camus's father, Lucien Camus, a poor agricultural worker, was killed in 1914 at the First Battle of the Marne, when Albert was an infant."),
("family","Camus's mother, Catherine Hélène Sintès, was of Spanish (Menorcan) descent, partially deaf, illiterate, and worked as a cleaning woman."),
("family","Camus grew up in poverty in the working-class Belcourt district of Algiers, raised by his mother and grandmother."),
("family","Camus had an older brother named Lucien."),
("family","Camus married Simone Hié in 1934; the marriage failed, in part due to her morphine addiction."),
("family","Camus married his second wife, Francine Faure, a pianist and mathematician, in 1940."),
("family","Camus and Francine Faure had twins, a daughter Catherine and a son Jean, born in 1945."),
("family","Camus had a long love affair with the Spanish-French actress Maria Casarès."),
("family","Camus had a cat named Cigarette; a devoted smoker, he gave the cat that pointed name. (Sartre's cat was named Néant, 'Nothingness'.)"),
# ── education / mentors ──
("education","Camus's primary-school teacher Louis Germain recognised his talent and helped him win a scholarship; Camus dedicated his Nobel lecture to him."),
("education","Camus's philosophy teacher and early mentor was Jean Grenier."),
("education","Camus studied philosophy at the University of Algiers; he did not attend the Sorbonne."),
("education","Camus's higher-studies thesis was on Plotinus and Saint Augustine — Neoplatonism and early Christian metaphysics."),
("health","Camus contracted tuberculosis at seventeen; it recurred throughout his life and ended both his teaching ambitions and his football."),
# ── football / theatre ──
("football","Camus played as a goalkeeper for the junior team of Racing Universitaire d'Alger (RUA) until tuberculosis stopped him at seventeen."),
("football","Camus said he learned his clearest lessons about morality and obligation from sport and the theatre, not the lecture hall."),
("theatre","Camus loved the theatre all his life and co-founded a troupe in Algiers, the Théâtre du Travail, later the Théâtre de l'Équipe."),
# ── career ──
("career","In the late 1930s Camus worked as a journalist at Alger Républicain, reporting on the poverty of the Kabylia region."),
("career","During the Second World War Camus edited Combat, the clandestine newspaper of the French Resistance."),
("career","Camus worked for years as an editor at the Gallimard publishing house in Paris."),
# ── works ──
("works","Camus wrote the novel The Stranger (L'Étranger), published in 1942."),
("works","Camus wrote the philosophical essay The Myth of Sisyphus (Le Mythe de Sisyphe), published in 1942."),
("works","Camus wrote the novel The Plague (La Peste), published in 1947."),
("works","Camus wrote the essay The Rebel (L'Homme révolté), published in 1951."),
("works","Camus wrote the novel The Fall (La Chute), published in 1956."),
("works","Camus wrote the short-story collection Exile and the Kingdom (L'Exil et le royaume), published in 1957."),
("works","Camus wrote the play Caligula, staged in 1945."),
("works","Camus wrote the play The Misunderstanding (Le Malentendu), 1944."),
("works","Camus wrote the play The Just (Les Justes), 1949, and State of Siege (L'État de siège), 1948."),
("works","Camus's early essay collections were Betwixt and Between (L'Envers et l'endroit, 1937) and Nuptials (Noces, 1938)."),
("works","Camus wrote Letters to a German Friend (Lettres à un ami allemand), 1945."),
("works","Camus's early novel A Happy Death (La Mort heureuse) was published posthumously in 1971."),
("works","Camus's unfinished autobiographical novel The First Man (Le Premier Homme) was found in the crash wreckage and published in 1994."),
("works","Camus's notebooks (Carnets) were published after his death."),
# ── nobel ──
("nobel","Camus was awarded the Nobel Prize in Literature in 1957; he was one of the youngest literature laureates."),
("nobel","Camus did not win the Nobel Prize in Physics; his Nobel was in Literature, in 1957."),
("nobel","Camus never won the Pulitzer Prize, which is an American award, nor the Prix Goncourt."),
("nobel","In his Nobel address Camus spoke of the artist's responsibility to truth and to human freedom, and he honoured his old teacher Louis Germain."),
# ── philosophy ──
("philosophy","Camus is associated with absurdism: the confrontation between the human demand for meaning and the silent indifference of the universe."),
("philosophy","Camus rejected the label 'existentialist' and distinguished his thought from that of Sartre."),
("philosophy","Camus's recurring themes are the absurd, revolt, freedom, measure (mesure), and human solidarity."),
("philosophy","The Myth of Sisyphus ends with the idea that one must imagine Sisyphus happy."),
("philosophy","Camus grouped his work in cycles: the absurd (The Stranger, Sisyphus, Caligula), then revolt (The Plague, The Rebel, The Just); a planned cycle on love was left unfinished."),
("philosophy","For Camus, revolt means refusing what crushes human dignity without the illusion of final victory — saying no while affirming a shared human worth."),
# ── politics / Sartre ──
("politics","Camus was briefly a member of the Communist Party in Algeria in the mid-1930s before leaving it."),
("politics","Camus and Jean-Paul Sartre were friends, then broke publicly in 1952 after the controversy over The Rebel."),
("politics","Camus opposed capital punishment and argued against it in his 1957 essay Reflections on the Guillotine."),
("politics","During the Algerian War Camus sought a civil truce between the communities and was deeply anguished by the conflict."),
("politics","Asked about Algeria in Stockholm in 1957, Camus made a much-discussed remark that he believed in justice but would defend his mother before an abstract justice."),
("politics","Camus opposed totalitarian ideologies of both the right and the left, defending a Mediterranean ethic of measure and limit."),
# ── personal / appearance ──
("personal","Camus smoked, and is often photographed with a cigarette and in a trench coat."),
("personal","Camus loved the sea, the sun, Algiers, the Roman ruins at Tipasa, and later the light of Provence."),
("personal","Camus took pride in his Spanish heritage through his mother and admired Spanish culture and theatre."),
# ── negative / common confusions ──
("correction","Camus was born in Algeria, not in Paris or mainland France."),
("correction","Camus did not write The Unbearable Lightness of Being (that is Milan Kundera) or The Labyrinth of Solitude (that is Octavio Paz)."),
("correction","Camus was never a professional athlete; he played amateur football as a youth goalkeeper only."),
("correction","Camus did not serve as a soldier in the war; tuberculosis kept him out, and he served the Resistance through journalism."),
# --- targeted fixes for generic-query retrieval misses + one confabulation ---
("family","Camus was fond of cats and kept them as pets throughout his life; among his cats was one named Cigarette."),
("education","Camus's primary schoolteacher was named Louis Germain."),
("education","Louis Germain, Camus's primary school teacher in Belcourt, recognised his talent and won him a scholarship to the lycée; Camus dedicated his 1957 Nobel acceptance speech to Germain."),
("life","Camus was born in Mondovi (now Dréan) in eastern Algeria in 1913 — not in Algiers. Algiers, in the Belcourt district, is where he was raised from infancy, but his birthplace was Mondovi."),
("personal","Camus was a writer, journalist, playwright, and moralist; he studied philosophy, not law, and was never a lawyer."),
("personal","Camus was a French-Algerian writer, journalist, playwright, and moralist, born in 1913 and raised in poverty in Algiers, known for The Stranger, The Plague, and The Fall and for his philosophy of the absurd and revolt."),
("personal","Camus had a cat named Cigarette; a devoted smoker, he gave the cat that pointed name. (Sartre's cat was named Néant, 'Nothingness'.)"),
("identity","Camus was a French-Algerian novelist, essayist, playwright and journalist, born in 1913 and awarded the Nobel Prize in Literature in 1957; he studied philosophy in Algiers but was never a lawyer."),
    ("teacher","Camus's primary-school teacher Louis Germain recognised the promise of a poor boy in Algiers and helped him win a scholarship; Camus dedicated his Nobel Prize to him and wrote him a grateful letter."),
    ("birthplace","Camus was born on 7 November 1913 in Mondovi (now Dréan) in French Algeria and grew up poor in the working-class Belcourt district of Algiers."),
    ("family","Camus's father Lucien died at the Battle of the Marne in 1914; his mother Catherine, partly deaf and unable to read, cleaned houses; he was raised by her and a severe grandmother in Algiers."),
    ("death","Camus died on 4 January 1960 in a car crash near Villeblevin, France, aged 46."),
 
    # --- works, by category (the anchor that fixes "name your works") ---
    ("works","Camus's best-known works are the novels The Stranger, The Plague and The Fall; the short-story collection Exile and the Kingdom; the essays The Myth of Sisyphus and The Rebel; and the plays Caligula and The Just."),
    ("works","Camus's novels are The Stranger (1942), The Plague (1947) and The Fall (1956); A Happy Death and the unfinished The First Man were published only after his death. The Rebel is NOT a novel — it is an essay."),
    ("works","Camus's one collection of short stories is Exile and the Kingdom (1957), six stories including 'The Guest'."),
    ("works","Camus's principal essays are The Myth of Sisyphus (1942) and The Rebel (1951); his early lyrical essay collections are The Wrong Side and the Right Side (also called Betwixt and Between, 1937), Nuptials (1939) and Summer (1954)."),
    ("works","Camus's plays are Caligula (1944), The Misunderstanding (1944), State of Siege (1948) and The Just / Les Justes (1949)."),
 
    # --- elevator pitches ---
    ("works","The Stranger (L'Étranger, 1942): Meursault, a detached French-Algerian clerk, shoots a man on a sunlit Algiers beach and is condemned as much for showing no grief at his mother's funeral as for the killing — a portrait of the absurd."),
    ("works","The Myth of Sisyphus (1942): a philosophical essay asking whether life is worth living in a world without inherent meaning; Camus rejects suicide and argues for lucid revolt, ending 'one must imagine Sisyphus happy.'"),
    ("works","The Plague (La Peste, 1947): when plague seals off the city of Oran, Dr Rieux and others fight it without hope of reward — an allegory of resistance to evil and of human solidarity against absurd suffering."),
    ("works","The Fall (La Chute, 1956): in an Amsterdam bar the former Paris lawyer Jean-Baptiste Clamence delivers an ironic confession that draws the listener into universal guilt and hypocrisy — Camus's darkest book."),
    ("works","The Rebel (L'Homme révolté, 1951): a long essay on metaphysical and historical rebellion, condemning revolutions that murder the living for an abstract future; its argument triggered Camus's break with Sartre."),
 
    # --- famous quotes, with context ---
    ("quotes","'One must imagine Sisyphus happy' is the closing line of The Myth of Sisyphus (1942): condemned to roll his rock forever, Sisyphus owns his fate, and the struggle itself is enough to fill a man's heart."),
    ("quotes","'In the midst of winter, I found there was, within me, an invincible summer' is from the essay 'Return to Tipasa' (1952), collected in Summer (L'Été) — an inner strength no 'winter' can destroy."),
 
    # --- misattribution debunk ---
    ("correction","The popular saying 'Don't walk in front of me, I may not follow; don't walk behind me, I may not lead; just walk beside me and be my friend' is NOT by Camus — he never wrote or said it; the 'Nobel Banquet 1957' citation attached to it is fabricated."),
 
    # --- Sartre feud ---
    ("life","Camus and Jean-Paul Sartre were friends from the early 1940s but broke publicly in 1952: after The Rebel (1951), Sartre's journal Les Temps Modernes ran a hostile review by Francis Jeanson; Sartre then wrote a scathing open letter ending the friendship. They never reconciled."),
 
    # --- philosophy labels ---
    ("philosophy","Camus rejected the label 'existentialist'; he said he and Sartre were surprised to see their names linked, and thought of himself as a writer of the absurd and of a 'moral classicism', not a member of any system."),
    ("philosophy","Camus did not believe in God and was not religious, yet he was not a militant atheist; he refused both religious consolation and nihilism, holding that one must create meaning and act justly without transcendence."),
    ("philosophy","Camus defined the absurd as the confrontation between the human longing for meaning and the silent, indifferent universe that offers none — answered not by suicide or false hope but by lucid revolt."),
 
    # --- political stances ---
    ("politics","On the Algerian War, Camus opposed both colonial injustice and FLN terrorism; in January 1956 he made a public Appeal for a Civilian Truce in Algiers, asking both sides to spare civilians, and was attacked by both camps."),
    ("politics","Camus's remark 'I believe in justice, but I will defend my mother before justice' was made at Stockholm in 1957: he meant that if 'justice' meant bombs in the trams of Algiers that might kill his mother, he could not call it justice — not a rejection of justice itself."),
    ("politics","Camus was an absolute opponent of the death penalty; his 1957 essay 'Reflections on the Guillotine' argues capital punishment is premeditated, useless as deterrence, and barbaric."),
    ("works",
  "What are your works? What did you write? Name your works. Camus's works are: the "
  "novels The Stranger (1942), The Plague (1947) and The Fall (1956); the short-story "
  "collection Exile and the Kingdom (1957); the essays The Myth of Sisyphus (1942) and "
  "The Rebel (1951); and the plays Caligula, The Misunderstanding, State of Siege and "
  "The Just. The Rebel is an essay, not a novel."),
  ("pets",
  "Camus loved animals and kept pets: cats — including one he named Cigarette — and, "
  "over the years, several dogs (among them Pauline, Kirk and Blaise).")
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="./camus_kb.jsonl"); a=ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True)
    with open(a.out,"w",encoding="utf-8") as f:
        for i,(topic,text) in enumerate(FACTS):
            f.write(json.dumps({"id":i,"topic":topic,"text":text},ensure_ascii=False)+"\n")
    from collections import Counter
    print(f"✅ wrote {len(FACTS)} facts -> {a.out}")
    print("   topics:", dict(Counter(t for t,_ in FACTS)))

if __name__=="__main__": main()

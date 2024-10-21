A partire dal lancio di Facebook nel 2004 i social media si sono evoluti e sono diventati parte integrante della quotidianità di un enorme fetta della popolazione.
Insieme a loro sono evoluti anche i sistemi di raccomandazione che popolano il feed di migliaia di utenti quotidianamente.

Secondo diversi studi, da cui parte questo approfondimento, l'eccessivo utilizzo dei social media può essere classificata come una vera e propria dipendenza comportamentale. Questo studio si pone l'obbiettivo di analizzare il ruolo dei recommender systems nello sviluppo di dipendenze comportamentali.

L'approccio scelto coinvolge la definizione di un ambiente multi-agente composto da: Utente e Recommender System che collaborano o competono tra loro.

In questo contesto si definisce la dipendenza come un errore del sistema di decision making dell'utente. 
L'utente è stato definito con un approccio basato sul Reinforcement Learning (RL), utilizzando in particolare un dual RL model system.
Questo doppio modello fa sì che le decisioni prese dal nostro utente siano il risultato sia delle sue abitudini(Model Free) che 
dalle sue capacità di planning e problem solving(Model Based).

Mentre il sistema di raccomandazione utilizza un multi-armed bandit per catturare le preferenze dell'utente e suggerire contenuti in base alle sue preferenze. 

L'ambiente in cui Utente e Recommender System operano è di fatto un grafo: i nodi rappresentano i vari stati psicofisici in cui l’utente si può trovare, mentre gli archi sono le azioni (dell'utente) necessarie per passare da uno stato all’altro. 

I risultati ottenuti mostrano che dual RL model system è in grado di emulare realisticamente il processo decisionale degli utenti. In particolare, si osserva che una conoscenza accurata dell’ambiente conduce a politiche ottimali, mentre quando il modello interno è costruito in modo errato, si osserva come la mancanza di conoscenza viene compensata dall’esperienza maturata (Model Free) nel corso delle iterazioni. 

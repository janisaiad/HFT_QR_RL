#import "@preview/typographix-polytechnique-reports:0.1.4" as template

// Defining variables for the cover page and PDF metadata
// Main title on cover page
#let title = [Rapport de stage en entreprise
#linebreak()
sur plusieurs lignes]
// Subtitle on cover page
#let subtitle = "Un sous-titre pour expliquer ce titre"
// Logo on cover page
#let logo = none // instead of none set to image("path/to/my-logo.png")
#let logo-horizontal = true // set to true if the logo is squared or horizontal, set to false if not
// Short title on headers
#let short-title = "Rapport de stage"
#let author = "Rémi Germe"
#let date-start = datetime(year: 2024, month: 06, day: 05)
#let date-end = datetime(year: 2024, month: 09, day: 05)
// Set to true for bigger margins and so on (good luck with your report)
#let despair-mode = false

#set text(lang: "fr")

// Set document metadata
#set document(title: title, author: author, date: datetime.today())
#show: template.apply.with(despair-mode: despair-mode)

// Cover page
#template.cover.cover(title, author, date-start, date-end, subtitle: subtitle, logo: logo, logo-horizontal: logo-horizontal)
#pagebreak()

// Acknowledgements
#heading(level: 1, numbering: none, outlined: false)[Remerciements]
#lorem(250)
#pagebreak()

// Executive summary
#heading(level: 1, numbering: none, outlined: false)[Executive summary]
#lorem(300)
#pagebreak()

// Table of contents
#outline(title: [Template contents], indent: 1em, depth: 2)

// Defining header and page numbering (will pagebreak)
#show: template.page.apply-header-footer.with(short-title: short-title)

// Introduction
#heading(level: 1, numbering: none)[Introduction]
#lorem(400)
#pagebreak()

// Here goes the main content

= Notations et concepts préliminaires

== Microstructure des marchés financiers
#lorem(50)

=== Carnet d'ordres et formation des prix
#lorem(120)

=== Mesures de liquidité
#lorem(80)

== Le modèle Queue-Reactive
#lorem(35)

=== Hypothèses principales
#lorem(100)

=== Prédictions théoriques
#lorem(90)

#pagebreak()






= Analyse empirique des données de Chicago




== Méthodologie de traitement des données

=== Nettoyage et préparation
#table(
  columns: (auto, auto, auto),
  [Critère], [Seuil], [Impact],
  [Outliers], [±3σ des prix], [Suppression des points aberrants],
  [Missing data], [>5% par jour], [Interpolation linéaire],
  [Valeurs aberrantes], [614 dans les tailles], [Remplacement par 0],
  [Sélection d'actif], ["LCID"], [Focus sur un titre unique],
  [Profondeur du carnet], [Niveau 0], [Analyse du BBO],
  [Publisher ID], [39], [Sélection des données de publisher_id 39],
  [Horodatage], [ts_event], [On ne garde que les événements entre 14h et 20h]
)
=== Traitement des données haute fréquence
Le processus de nettoyage des données de carnet d'ordres suit plusieurs étapes critiques pour assurer la qualité de l'analyse :

==== Gestion des valeurs aberrantes
Une attention particulière est portée aux valeurs aberrantes dans les tailles d'ordres. La valeur spécifique 614 a été identifiée comme un marqueur d'erreur ou de données manquantes dans le flux de données. Ces occurrences sont systématiquement remplacées par des zéros pour éviter toute distorsion dans l'analyse quantitative.

==== Filtrage des données
Le processus de filtrage s'effectue selon plusieurs dimensions :
- *Sélection de la source* : Utilisation exclusive des données du publisher_id 39
- *Focus sur un instrument* : Analyse centrée sur le titre "LCID"
- *Profondeur du carnet* : Concentration sur le meilleur niveau de prix (depth = 0)

===== Données par publisher_id
#table(
  columns: (auto, auto),
  [Publisher ID], [Nombre de points],
  [39], [1 000 000], [39], [1 000 000]
)

==== Calcul des variations
Pour capturer la dynamique du carnet d'ordres, des différences premières sont calculées sur les séries temporelles des tailles d'ordres :
- Variations des tailles d'ordres à l'achat (bid_size_diff)
- Variations des tailles d'ordres à la vente (ask_size_diff)
Ces métriques permettent d'observer les changements instantanés dans la liquidité du marché.

==== Validation et contrôle qualité
Un système de validation rigoureux a été mis en place pour vérifier la cohérence des modifications du carnet d'ordres :

===== Typologie des actions
Trois types d'actions sont validés selon des règles spécifiques :
- *Transactions (T)* : Vérification que la diminution de la taille correspond exactement au volume échangé
- *Ajouts (A)* : Confirmation que l'augmentation de la taille correspond à l'ordre ajouté
- *Annulations (C)* : Validation que la diminution de la taille correspond à l'ordre annulé

===== Système de notation
Un système binaire de validation a été implémenté :
- Status "OK" : L'action respecte les règles de cohérence
- Status "NOK" : L'action présente une anomalie

Cette validation permet d'identifier les incohérences potentielles dans les données et d'assurer la fiabilité des analyses subséquentes.

===== Traitement technique
Les étapes techniques incluent :
- Conversion des types de données (cast en Int64) pour supporter les valeurs négatives
- Vérification systématique de la présence des colonnes requises
- Création d'indicateurs de qualité (status_N et status_diff) pour le suivi des anomalies

==== Analyse temporelle des événements
L'analyse de la dimension temporelle des données de marché nécessite un traitement spécifique :

===== Normalisation temporelle
Le processus de normalisation temporelle comprend plusieurs étapes :
- *Conversion des timestamps* : Les événements sont convertis en format datetime standardisé
- *Indexation séquentielle* : Attribution d'indices séquentiels pour suivre l'ordre des événements
- *Calcul des intervalles* : Mesure des écarts temporels entre événements consécutifs en nanosecondes

===== Filtrage temporel
Un filtrage temporel est appliqué pour éliminer les anomalies :
- Suppression des événements simultanés (intervalle de temps nul) en grands nombre
- Identification des sauts temporels anormaux
- Conservation uniquement des événements avec des intervalles temporels significatifs

===== Métriques dérivées
Plusieurs métriques temporelles sont calculées :
- *Temps écoulé* : Mesure en secondes entre événements consécutifs
- *Différences d'indices* : Identification des discontinuités dans la séquence d'événements
- *Fréquence d'événements* : Analyse de la densité temporelle des modifications du carnet d'ordres

==== Filtrage conditionnel des événements
Une série de filtres conditionnels est appliquée pour assurer la cohérence des données :

===== Filtres sur les modifications du carnet
#table(
  columns: (auto, auto),
  [Type d'événement], [Condition de validation],
  [Annulations (C)], [Variation de taille non nulle],
  [Ajouts (A)], [Variation = taille de l'ordre],
  [Transactions (T)], [Variation = -taille de l'ordre]
)

===== Traitement des cas particuliers
- Élimination des annulations sans impact sur le carnet
- Validation des ajouts avec correspondance exacte des tailles
- Vérification de la cohérence des transactions avec les variations observées

==== Statistiques descriptives sur le filtrage



Cette approche méthodologique garantit une base de données propre et cohérente pour les analyses ultérieures, en particulier pour l'étude de la microstructure du marché et de la formation des prix.





=== Construction des métriques
Les données de trading haute fréquence sont agrégées en barres de volume et de temps pour extraire des métriques pertinentes comme les imbalances d'ordres, la profondeur du carnet d'ordres et la volatilité réalisée. Une attention particulière est portée à la normalisation des données pour tenir compte des effets saisonniers intra-journaliers.

== Faits stylisés observés

=== Analyse des imbalances
Les déséquilibres entre ordres d'achat et de vente montrent une forte autocorrélation à court terme et des clusters de volatilité caractéristiques des marchés financiers. La distribution des imbalances présente des queues épaisses typiques des données haute fréquence.

// #figure(
  // caption: "Distribution des imbalances sur le marché de Chicago"
// )

=== Comportement des providers
#table(
  columns: (auto, auto, auto),
  [Provider], [Caractéristique], [Statistique],
  [Market Makers], [Ratio ordre/trade], [15.3],
  [HFT Firms], [Temps de latence moyen], [< 10ms]
)


#pagebreak()
== Comparaison avec le modèle Queue-Reactive

=== Adéquation aux prédictions théoriques
Les données empiriques confirment largement les prédictions du modèle Queue-Reactive, notamment concernant la formation des prix et la dynamique des carnets d'ordres. Les temps d'arrivée des ordres suivent une distribution de Poisson modifiée, comme prévu par le modèle théorique.

=== Cas particuliers et anomalies
Certains événements de marché comme les annonces macroéconomiques ou les périodes de forte volatilité montrent des déviations par rapport au modèle. Ces anomalies se caractérisent par des pics de volume anormaux et une augmentation temporaire de la corrélation entre les différentes métriques de liquidité.



==== Anomalies

Quand il y a un trade, on a un cancel sur la bonne queue 
et sur la mauvaise queue on a un trade

















#pagebreak()


// Conclusion
#heading(level: 1, numbering: none)[Conclusion]
#lorem(200)

// Bibliography (if necessary)
// #pagebreak()
// #bibliography("path-to-file.bib")




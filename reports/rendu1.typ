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
  [Missing data], [>5% par jour], [Interpolation linéaire]
)

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
  [HFT Firms], [Temps de latence moyen], [<10ms]
)

== Comparaison avec le modèle Queue-Reactive

=== Adéquation aux prédictions théoriques
Les données empiriques confirment largement les prédictions du modèle Queue-Reactive, notamment concernant la formation des prix et la dynamique des carnets d'ordres. Les temps d'arrivée des ordres suivent une distribution de Poisson modifiée, comme prévu par le modèle théorique.

=== Cas particuliers et anomalies
Certains événements de marché comme les annonces macroéconomiques ou les périodes de forte volatilité montrent des déviations par rapport au modèle. Ces anomalies se caractérisent par des pics de volume anormaux et une augmentation temporaire de la corrélation entre les différentes métriques de liquidité.





















#pagebreak()


// Conclusion
#heading(level: 1, numbering: none)[Conclusion]
#lorem(200)

// Bibliography (if necessary)
// #pagebreak()
// #bibliography("path-to-file.bib")




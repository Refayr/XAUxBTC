//-------------------------------------
// Document options
//
#let option = (
  //type : "final",
  type: "draft",
  lang: "en",
  //lang : "de",
  //lang : "fr",
)
//-------------------------------------
// Optional generate titlepage image
//
#import "@preview/fractusist:0.1.1": *  // only for the generated images

#let titlepage_logo = dragon-curve(
  12,
  step-size: 10,
  stroke-style: stroke(
    //paint: gradient.linear(..color.map.rocket, angle: 135deg),
    paint: gradient.radial(..color.map.rocket),
    thickness: 3pt,
    join: "round",
  ),
  height: 10cm,
)

//-------------------------------------
// Metadata of the document
//
#let doc = (
  title: [*Gold and cryptocurrencies exchange rates correlation*],
  abbr: "Prj",
  subtitle: [_Python Project_],
  url: "https://github.com/Refayr/XAUxBTC",
  logos: (
    tp_topleft: image("resources/img/ipparis.jpg", height: 1.2cm),
    tp_topright: image("resources/img/polytechnique.jpg", height: 1.5cm),
    tp_main: titlepage_logo,
    header: image("resources/img/project-logo.svg", width: 2.5cm),
  ),
  authors: (
    (
      name: "Zhousiying Wu",
      abbr: "Ruby",
      email: "zhousiying.wu@polytechnique.edu",
      //url: "https://synd.hevs.io",
    ),
    (
      name: "Nicolas Valety",
      abbr: "NV",
      email: "nicolas.valety@ip-paris.fr",
      url: "https://github.com/Refayr/XAUxBTC",
    ),
  ),
  school: (
    name: "Polytechnique Institute of Paris",
    major: "Applied Mathematics and Statistics",
    orientation: "Master degree",
    url: "https://www.polytechnique.edu/",
  ),
  course: (
    name: "Python for Datascience",
    url: "https://github.com/AnasHattay/",
    prof: "Anas Hattay",
    class: [M1 APPMS],
    semester: "Fall Semester 2025",
  ),
  keywords: ("Typst", "Template", "Report", "HEI-Vs", "Project", "Gold", "Crypto"),
  version: "v0.1.0",
)

#let date = datetime.today()

//-------------------------------------
// Settings
//
#let tableof = (
  toc: true,
  tof: false,
  tot: false,
  tol: false,
  toe: false,
  maxdepth: 3,
)

#let gloss = true
#let appendix = false
#let bib = (
  display: true,
  path: "/tail/bibliography.bib",
  style: "ieee", //"apa", "chicago-author-date", "chicago-notes", "mla"
)

# Repository Coverage



| Name                                                                |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| \_\_init\_\_.py                                                     |        0 |        0 |        0 |        0 |    100% |           |
| protzilla/\_\_init\_\_.py                                           |        0 |        0 |        0 |        0 |    100% |           |
| protzilla/constants/\_\_init\_\_.py                                 |        0 |        0 |        0 |        0 |    100% |           |
| protzilla/constants/colors.py                                       |        2 |        0 |        0 |        0 |    100% |           |
| protzilla/constants/location\_mapping.py                            |        7 |        0 |        2 |        0 |    100% |           |
| protzilla/constants/paths.py                                        |        8 |        0 |        0 |        0 |    100% |           |
| protzilla/constants/protzilla\_logging.py                           |       42 |       10 |       10 |        0 |     62% |46-47, 50-51, 54-55, 58-59, 62-63 |
| protzilla/data\_analysis/\_\_init\_\_.py                            |        0 |        0 |        0 |        0 |    100% |           |
| protzilla/data\_analysis/classification.py                          |       65 |       32 |       12 |        4 |     48% |33-52, 54, 56-66, 67->exit, 263-313 |
| protzilla/data\_analysis/classification\_helper.py                  |       82 |       15 |       40 |        5 |     74% |41, 81-82, 104-115, 132, 167-170 |
| protzilla/data\_analysis/clustering.py                              |       69 |       11 |        6 |        2 |     83% |141, 365-380, 384-388 |
| protzilla/data\_analysis/differential\_expression.py                |        7 |        3 |        0 |        0 |     57% |      9-11 |
| protzilla/data\_analysis/differential\_expression\_anova.py         |       58 |       14 |       30 |        7 |     69% |70-72, 78->91, 81-87, 102-107, 110, 127-128, 139 |
| protzilla/data\_analysis/differential\_expression\_helper.py        |       27 |        2 |       10 |        2 |     89% |    43, 60 |
| protzilla/data\_analysis/differential\_expression\_linear\_model.py |       66 |       15 |       24 |        6 |     70% |55-56, 63-64, 82-88, 91, 131-137 |
| protzilla/data\_analysis/differential\_expression\_t\_test.py       |       64 |        2 |       24 |        2 |     95% |20, 75->79, 106 |
| protzilla/data\_analysis/dimension\_reduction.py                    |       33 |        4 |       10 |        2 |     86% |67-72, 100, 179 |
| protzilla/data\_analysis/model\_evaluation.py                       |       10 |        0 |        0 |        0 |    100% |           |
| protzilla/data\_analysis/model\_evaluation\_plots.py                |       18 |        0 |        0 |        0 |    100% |           |
| protzilla/data\_analysis/plots.py                                   |      131 |       41 |       64 |        5 |     63% |75, 84->86, 137, 282, 312-319, 325-461 |
| protzilla/data\_analysis/protein\_graphs.py                         |      409 |       46 |      196 |       15 |     89% |32-40, 63-92, 148, 173-174, 188, 319-321, 330-332, 350-351, 400, 425-429, 473, 487, 556, 587, 594, 795, 853->852, 857-860 |
| protzilla/data\_integration/\_\_init\_\_.py                         |        0 |        0 |        0 |        0 |    100% |           |
| protzilla/data\_integration/database\_download.py                   |       63 |       63 |       28 |        0 |      0% |     1-116 |
| protzilla/data\_integration/database\_integration.py                |       62 |       14 |       36 |        1 |     79% |73, 102-137 |
| protzilla/data\_integration/database\_query.py                      |      115 |       44 |       54 |        7 |     59% |27-80, 84-86, 90, 103->exit, 104->103, 107->104, 111-112, 116-123, 149->148, 161-166, 167->176, 177-180 |
| protzilla/data\_integration/di\_plots.py                            |      124 |       21 |       58 |       11 |     82% |77, 106->109, 110, 176-177, 180-181, 183, 186-187, 215-217, 237-242, 295-296, 299, 373 |
| protzilla/data\_integration/enrichment\_analysis.py                 |      338 |       94 |      166 |       39 |     71% |24-25, 49->51, 178-179, 194->200, 204->210, 211-213, 215, 222, 251-255, 277-281, 329, 392-395, 410-411, 426-427, 533-536, 545-546, 551-553, 557->578, 562, 565-571, 584-585, 597-607, 609->619, 611-612, 614-617, 619->632, 629-630, 633-643, 646-647, 663, 744-745, 754-755, 768-769, 771-772, 774-777, 781-782, 784-787, 800, 818-819, 831-832, 835-836, 846-849 |
| protzilla/data\_integration/enrichment\_analysis\_gsea.py           |      132 |        4 |       54 |        6 |     95% |155, 158, 222->226, 387, 390, 461->465 |
| protzilla/data\_integration/enrichment\_analysis\_helper.py         |       73 |        6 |       42 |        2 |     93% |137-139, 145, 150-151 |
| protzilla/data\_preprocessing/\_\_init\_\_.py                       |        0 |        0 |        0 |        0 |    100% |           |
| protzilla/data\_preprocessing/filter\_proteins.py                   |       18 |        2 |        4 |        1 |     77% |     53-54 |
| protzilla/data\_preprocessing/filter\_samples.py                    |       35 |        0 |        4 |        0 |    100% |           |
| protzilla/data\_preprocessing/imputation.py                         |      131 |        3 |       30 |        4 |     96% |47-53, 180, 517->530, 541->547 |
| protzilla/data\_preprocessing/normalisation.py                      |       98 |        1 |       22 |        2 |     98% |248->259, 260 |
| protzilla/data\_preprocessing/outlier\_detection.py                 |       67 |        3 |       12 |        4 |     89% |176, 193, 248, 249->exit |
| protzilla/data\_preprocessing/peptide\_filter.py                    |       16 |        2 |        4 |        1 |     75% |     50-51 |
| protzilla/data\_preprocessing/plots.py                              |      101 |       11 |       23 |        4 |     88% |165->180, 249-250, 284, 295, 395-419 |
| protzilla/data\_preprocessing/plots\_helper.py                      |       17 |       13 |        6 |        0 |     17% |15-24, 38-47 |
| protzilla/data\_preprocessing/transformation.py                     |       21 |        2 |        8 |        3 |     83% |33, 42->52, 53 |
| protzilla/history.py                                                |      135 |       10 |       60 |        6 |     92% |40, 124, 134, 180-184, 227, 234 |
| protzilla/importing/\_\_init\_\_.py                                 |        0 |        0 |        0 |        0 |    100% |           |
| protzilla/importing/metadata\_import.py                             |       64 |       15 |       28 |        7 |     76% |29, 31, 45-47, 91-98, 113, 124, 129-138, 168-169 |
| protzilla/importing/ms\_data\_import.py                             |      115 |       22 |       48 |        1 |     79% |28-29, 89-91, 119-121, 233-251 |
| protzilla/importing/peptide\_import.py                              |       22 |        2 |        2 |        0 |     92% |     15-16 |
| protzilla/run.py                                                    |      343 |       49 |      134 |       10 |     84% |76-82, 86-92, 170->181, 202, 204, 239-243, 292-298, 308, 323-336, 375-376, 413-416, 467, 507, 511-514, 533, 545->518 |
| protzilla/run\_helper.py                                            |       94 |       22 |       78 |       12 |     73% |28, 47-48, 53, 58, 69, 76, 88, 92-97, 99-101, 104-106, 111, 126, 164->exit, 170-176, 178 |
| protzilla/runner.py                                                 |       93 |        4 |       38 |        4 |     94% |121, 130, 148->146, 162-163 |
| protzilla/utilities/\_\_init\_\_.py                                 |        1 |        0 |        0 |        0 |    100% |           |
| protzilla/utilities/clustergram.py                                  |      375 |       99 |      194 |       55 |     67% |82, 97, 99, 106, 150-151, 153, 155, 161->164, 164->171, 190, 205, 209, 213, 217, 227, 231-236, 241->243, 244, 246, 248, 252->254, 254->258, 259-270, 273, 275, 277-296, 315-318, 321->323, 328->330, 331, 383-384, 386-387, 402-403, 405-406, 474->480, 486, 503, 523, 526->549, 531->534, 568->574, 574->587, 662->667, 675->679, 696, 698, 728-735, 744-757, 767-771, 825->842, 842->862, 894->902, 902->911, 929-941, 944-956, 982-998, 1012-1028 |
| protzilla/utilities/dunn\_score.py                                  |       10 |        6 |        6 |        0 |     25% | 25, 41-48 |
| protzilla/utilities/transform\_dfs.py                               |       25 |        0 |        9 |        0 |    100% |           |
| protzilla/utilities/utilities.py                                    |       44 |        4 |       12 |        2 |     89% |27-28, 69, 80 |
| protzilla/workflow\_helper.py                                       |       58 |        0 |       36 |        0 |    100% |           |
| runner\_cli.py                                                      |       21 |        5 |        2 |        1 |     74% | 55-58, 62 |
| ui/\_\_init\_\_.py                                                  |        0 |        0 |        0 |        0 |    100% |           |
| ui/\_\_main\_\_.py                                                  |       11 |       11 |        2 |        0 |      0% |      2-21 |
| ui/main/\_\_init\_\_.py                                             |        0 |        0 |        0 |        0 |    100% |           |
| ui/main/asgi.py                                                     |        4 |        4 |        0 |        0 |      0% |     10-16 |
| ui/main/settings.py                                                 |       28 |        0 |        2 |        1 |     97% |    24->32 |
| ui/main/upload\_handler.py                                          |       37 |       37 |        2 |        0 |      0% |      1-73 |
| ui/main/urls.py                                                     |        4 |        4 |        0 |        0 |      0% |     16-21 |
| ui/main/views.py                                                    |       76 |       76 |       32 |        0 |      0% |     1-115 |
| ui/main/wsgi.py                                                     |        4 |        4 |        0 |        0 |      0% |     10-16 |
| ui/manage.py                                                        |       14 |       14 |        2 |        0 |      0% |      2-24 |
| ui/runs/\_\_init\_\_.py                                             |        0 |        0 |        0 |        0 |    100% |           |
| ui/runs/apps.py                                                     |        4 |        4 |        0 |        0 |      0% |       1-6 |
| ui/runs/fields.py                                                   |      128 |      108 |       76 |        0 |     10% |35-46, 64-97, 122-132, 145-147, 172-183, 197-202, 223-303, 320-345 |
| ui/runs/migrations/\_\_init\_\_.py                                  |        0 |        0 |        0 |        0 |    100% |           |
| ui/runs/templatetags/\_\_init\_\_.py                                |        0 |        0 |        0 |        0 |    100% |           |
| ui/runs/templatetags/id\_tags.py                                    |       10 |       10 |        4 |        0 |      0% |      1-13 |
| ui/runs/urls.py                                                     |        4 |        4 |        0 |        0 |      0% |       1-6 |
| ui/runs/utilities/\_\_init\_\_.py                                   |        0 |        0 |        0 |        0 |    100% |           |
| ui/runs/utilities/alert.py                                          |        2 |        1 |        0 |        0 |     50% |         2 |
| ui/runs/views.py                                                    |      344 |      286 |      130 |        1 |     13% |62, 87-134, 176-192, 223-244, 264-410, 423-430, 444-447, 462-466, 481-483, 498-507, 522-529, 544-552, 568-577, 595-604, 620-622, 635-639, 655-656, 703-705, 721-736, 744-774, 793-808, 814-866 |
| ui/runs/views\_helper.py                                            |       70 |       20 |       30 |        4 |     68% |16->18, 22, 35, 38-44, 116-130, 145-146, 155-158 |
|                                                           **TOTAL** | **4649** | **1299** | **1906** |  **239** | **69%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://github.com/cschlaffner/PROTzilla2/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/cschlaffner/PROTzilla2/tree/python-coverage-comment-action-data)

This is the one to use if your repository is private or if you don't want to customize anything.



## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.
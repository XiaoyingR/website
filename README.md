# SMAMI
Social Media Analytics for Municipal Issues

This project is in collaboration with Singapore's [Municipal Services Office (MSO)](www.mnd.gov.sg/mso/).
This project aims at extract municipal-related issues from multiple social media (Twitter, Facebook, Forum, Blogs, Alternative Media). A reference of municipal issues can be found [here](https://drive.google.com/a/smu.edu.sg/file/d/0B6Yi6-z5BVZNTzgzNGhPTzV3R1E/view?usp=sharing).

## Starting From Client's Perspective
To understand the client's perspective, we have to know that MSO is a joint office by different government agencies, and this project is aiming at improving its case-solving efficiency. A general procedure for agencies to deal with incoming complaints(thus issues) is to create a "case" - where issues could be coming from differet channels, such as email/phone call/message/online feedback etc.

The case is stored in a centralized database, and officers check respectively to decide whether to take actions on them. If yes, the officer contacts concerning agency and when receive feedback for that particular record, a case is considered being taken action and thus "closed".

The agency is faced with mainly 2 difficulties in their current system:
1. No particular channel is dedicated for **social media monitoring**. The case creation methodology is always passive and client want to take an active approach to detect such potential resources on social media.
2. Government's **intranet** prevents any users from accessing internet. They need to take an extra step to go to their own devices if any references in the case contains web resources (i.e. a url link).

## First-Phase Implementation
### KPI
We agreed that 2 criteria shall be meet:
1. Accuracy ~70% across categories
2. Location Indication ~10%
### Client's Objective
From what we know, the first-phase system would be tested in one to three agencies for demonstration purpose, highly possible agencies are LTA, SPF, HDB, NEA where most issues are categorized in. A mapping between agency and issues can be found [here](https://github.com/LARC-CMU-SMU/SMAMI/blob/master/Category-issue%20mapping.xlsx).
### System Overview
The system consists of:
* input: database ElasticSearch
* Processing: a trained issue classifier + a rule-based location detector & predictor
* Output: database ElasticSearch *mso_testing_v2* + web interface *MSO_web* + telegram push

A flowchart:

![diagram](https://github.com/LARC-CMU-SMU/SMAMI/blob/master/SMAMI_Diagram.png)

### Dive into details
#### [Location Detector and Predictor](https://github.com/LARC-CMU-SMU/SMAMI/tree/master/Location%20detector%20and%20predictor)
#### [Web Interface](https://github.com/LARC-CMU-SMU/SMAMI/tree/master/Web%20Interface)
#### [Telegram Bot](https://github.com/LARC-CMU-SMU/SMAMI/tree/master/Telegram-ES)

#### [Classifier Model Training](https://github.com/LARC-CMU-SMU/SMAMI/tree/master/Classifier%20Model%20Training)

------this part is left for jingjing to complete------

### Other Deveolped Packages that maybe useful
* [TextPreViz](https://github.com/yang0339/TextPreViz): text preprocessing + visualization (histogram + wordcloud)
* [GoogleSheetClient](https://github.com/yang0339/GoogleSheetClient): download workbook from Google Sheet
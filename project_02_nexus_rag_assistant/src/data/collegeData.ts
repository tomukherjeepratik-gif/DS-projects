import type { DocumentItem } from '../types/rag';

export const INITIAL_DOCUMENTS: DocumentItem[] = [
  {
    id: 'courses_and_syllabus',
    title: 'Courses & Syllabus',
    category: 'Courses',
    iconName: 'BookOpen',
    lastUpdated: '2026-09-15',
    content: `# Courses and Syllabus - Nexus Institute of Technology (NIT)

## 1. Undergraduate Programs (B.Tech)

### 1.1 B.Tech in Computer Science & Engineering (CSE)
- **Program Duration**: 4 Years (8 Semesters)
- **Total Credits Required**: 160 Credits
- **Department Head**: Dr. Elena Vance
- **Key Courses & Syllabus Breakdown**:
  - **CS101: Introduction to Computer Programming** (Semester 1 | 4 Credits)
    - Topics: Python fundamentals, memory management, data types, control structures, recursion, file I/O, basic data structures.
    - Prerequisites: None.
  - **CS204: Data Structures & Algorithms** (Semester 3 | 4 Credits)
    - Topics: Arrays, Linked Lists, Stacks, Queues, Binary Search Trees, AVL Trees, Heaps, Graph algorithms (Dijkstra, BFS, DFS), Sorting, Complexity analysis (Big-O).
    - Prerequisites: CS101. Instructor: Prof. Marcus Sterling.
  - **AI301: Fundamentals of Artificial Intelligence & Machine Learning** (Semester 5 | 4 Credits)
    - Topics: Supervised & unsupervised learning, linear regression, decision trees, neural networks, loss functions, optimization (SGD), evaluation metrics.
    - Prerequisites: CS204, MA201 (Linear Algebra & Probability).
  - **AI405: Large Language Models & Retrieval-Augmented Generation (RAG)** (Semester 7 | 4 Credits)
    - Topics: Transformer architecture, self-attention, tokenization, dense vector embeddings, vector databases (Faiss, Chroma), semantic chunking, prompt engineering, RAG pipeline design, evaluation metrics (RAGAS, BLEU, ROUGE).
    - Prerequisites: AI301. Instructor: Dr. Elena Vance.

### 1.2 B.Tech in Data Science & Analytics (DSA)
- **Program Duration**: 4 Years (8 Semesters)
- **Total Credits Required**: 158 Credits
- **Key Courses**:
  - **DS201: Applied Statistics & Probability** (Semester 3 | 3 Credits)
  - **DS305: Big Data Processing & Distributed Systems** (Semester 6 | 4 Credits) - Spark, Hadoop, SQL/NoSQL databases.

---

## 2. Postgraduate Programs (M.Tech)

### 2.1 M.Tech in Autonomous Systems & AI
- **Program Duration**: 2 Years (4 Semesters)
- **Total Credits Required**: 80 Credits
- **Core Course - AS501: Deep Learning for Robotics & Autonomous Systems** (Semester 2 | 4 Credits)
  - Focus: Computer vision, SLAM, reinforcement learning for robotics, edge AI inference.

---

## 3. Academic Grading & Honors Criteria

### 3.1 Grading Scale
- **A+ Grade (10 Points)**: 90% - 100% (Outstanding)
- **A Grade (9 Points)**: 80% - 89% (Excellent)
- **B Grade (8 Points)**: 70% - 79% (Very Good)
- **C Grade (7 Points)**: 60% - 69% (Good)
- **D Grade (6 Points)**: 50% - 59% (Pass)
- **F Grade (0 Points)**: Below 50% (Fail / Backlog)

### 3.2 B.Tech Honors Degree Requirements
- Students must maintain a minimum Cumulative Grade Point Average (CGPA) of **8.50 out of 10.0** at the end of Semester 6.
- Zero active backlogs or disciplinary warnings.
- Complete 12 additional credits from advanced elective courses (including AI405 or AS501).`
  },
  {
    id: 'fees_and_scholarships',
    title: 'Fees & Scholarships',
    category: 'Fees',
    iconName: 'DollarSign',
    lastUpdated: '2026-09-20',
    content: `# Tuition Fees, Financial Aid & Scholarships - Nexus Institute of Technology (NIT)

## 1. Fee Structure (Academic Year 2026-2027)

### 1.1 Undergraduate Programs (B.Tech)
- **Tuition Fee**: $4,500 USD (or ₹1,25,000 INR) per semester.
- **Development & Laboratory Fee**: $350 USD (₹10,000 INR) per semester.
- **Library & Technology Fee**: $150 USD (₹4,500 INR) per semester.
- **Refundable Caution Deposit**: $300 USD (₹8,500 INR) one-time admission deposit.

### 1.2 Hostel & Dining Charges
- **Standard Twin-Sharing Hostel Room**: $1,200 USD (₹35,000 INR) per semester (includes air conditioning, Wi-Fi, 24/7 security).
- **Single Occupancy Premium Room**: $1,800 USD (₹52,000 INR) per semester.
- **Mess & Dining Fee**: $600 USD (₹18,000 INR) per semester (4 meals daily, vegetarian and non-vegetarian options).

---

## 2. Payment Deadlines & Late Fine Policy
- **Fall Semester Due Date**: August 15, 2026.
- **Spring Semester Due Date**: January 15, 2027.
- **Late Payment Penalty**: $50 USD (₹1,500 INR) per week for late payments up to 3 weeks. After 3 weeks of non-payment, portal access is restricted and exam hall tickets are withheld.

---

## 3. Merit & Need-Based Scholarships

### 3.1 Academic Excellence Scholarship
- **Coverage**: 100% Tuition Fee Waiver for 4 years.
- **Eligibility**: Top 2% students in the entrance examination or candidates maintaining a CGPA of **9.50 or higher**.
- **Renewal Criteria**: CGPA must not drop below 9.00 in any semester.

### 3.2 Dean's Merit Scholarship
- **Coverage**: 50% Tuition Fee Waiver.
- **Eligibility**: Students with a CGPA between **9.00 and 9.49**.

### 3.3 Women in STEM Scholarship
- **Coverage**: 30% Tuition Fee Waiver.
- **Eligibility**: Female students enrolled in B.Tech Computer Science, Data Science, or Robotics maintaining CGPA >= 8.00.
- **Application Portal**: Apply via the student portal under \`Scholarships > STEM-Women\` by September 1 each year.

### 3.4 Need-Based Financial Aid
- **Coverage**: Up to 60% Tuition Assistance.
- **Eligibility**: Family annual income under $10,000 USD (₹8,00,000 INR). Income tax returns required.

---

## 4. Refund Policy
- **Withdrawal prior to Semester Start**: 100% refund of tuition fee (less $50 processing charge).
- **Withdrawal within 14 days of classes**: 80% refund of tuition fee.
- **Withdrawal between 15 to 30 days**: 50% refund of tuition fee.
- **Withdrawal after 30 days**: No tuition fee refund applicable. Caution deposit refunded in full.`
  },
  {
    id: 'clubs_and_societies',
    title: 'Clubs & Societies',
    category: 'Clubs',
    iconName: 'Users',
    lastUpdated: '2026-09-10',
    content: `# Campus Clubs & Student Societies - Nexus Institute of Technology (NIT)

## 1. Technical Clubs & Student Chapters

### 1.1 ACM Student Chapter (Association for Computing Machinery)
- **Faculty Sponsor**: Prof. Aris Thorne (Office: Building A - Room 415, Email: athorne@nexus.edu)
- **Student Lead / President**: Alex Rivera (Final Year B.Tech CSE)
- **Vice President**: Maya Lin (Third Year B.Tech Data Science)
- **Meeting Schedule**: Every **Wednesday at 5:00 PM** in Computer Lab 304 (Building A).
- **Core Activities & Flagship Events**:
  - Weekly Competitive Programming Contests ("Algo Wars" on HackerRank/Codeforces).
  - Annual 36-Hour Hackathon: **NexusHacks** (Held every February with cash prizes worth $5,000 USD).
  - Specialized Hands-on Workshops: RAG Architectures, Vector Search, LLM Fine-tuning, System Design, and Open Source Software.
- **Membership**: Open to all NIT students. Registration fee: $10/year (includes access to ACM Digital Library & cloud compute credits).

### 1.2 Robotics & Autonomous Systems Club ("RoboNexus")
- **Faculty Sponsor**: Dr. Vikram Patel
- **Meeting Schedule**: Every **Tuesday at 4:30 PM** in Innovation Lab 102 (Building C).
- **Projects**: Autonomous drone navigation, quadrupeds, Mars rover prototype for University Rover Challenge (URC).

### 1.3 CyberSec Guild (Ethical Hacking & Security)
- **Faculty Sponsor**: Dr. Sophia Chen
- **Meeting Schedule**: Every **Thursday at 5:30 PM** in Cyber Range Lab 201.
- **Focus Areas**: Capture The Flag (CTF) competitions, web application penetration testing, network security, cryptography.

---

## 2. Cultural & Creative Societies

### 2.1 "Aura" Cultural & Arts Society
- **Head Coordinator**: Sarah Jenkins
- **Annual Fest**: **Aetheria Fest** (Held in March over 3 days featuring battle of the bands, dance, fashion show, and guest concerts).
- **Rehearsal Location**: Student Activity Center (SAC) Auditorium.

### 2.2 Debating & Literary Society ("Nexus Dialogues")
- **Meeting Schedule**: Every **Friday at 4:00 PM** in Seminar Hall B.
- **Activities**: Parliamentary debates, Model United Nations (MUN), creative writing jams.

---

## 3. Sports & Esports Association
- **Facilities**: Outdoor Sports Complex (Football ground, 400m synthetic track, Floodlit Basketball courts) and Indoor Sports Arena (Badminton, Table Tennis, Chess).
- **NIT Esports League**: Seasonal tournaments in Valorant, Rocket League, and Chess.com blitz.`
  },
  {
    id: 'timetables_and_calendar',
    title: 'Timetables & Calendar',
    category: 'Timetables',
    iconName: 'Calendar',
    lastUpdated: '2026-09-18',
    content: `# Timetables & Academic Calendar - Nexus Institute of Technology (NIT)

## 1. Daily Lecture & Lab Schedule

### 1.1 Time Slots
- **Period 1**: 09:00 AM – 10:15 AM
- **Period 2**: 10:30 AM – 11:45 AM
- **Lunch & Break Hour**: 11:45 AM – 01:00 PM
- **Period 3**: 01:00 PM – 02:15 PM
- **Period 4**: 02:30 PM – 03:45 PM
- **Practical Lab Sessions**: 02:30 PM – 05:30 PM (3-hour block)

### 1.2 B.Tech CSE Semester 5 Sample Weekly Schedule
- **Monday**:
  - 09:00 AM – 10:15 AM: CS204 Data Structures (Lecture - Hall A1)
  - 10:30 AM – 11:45 AM: AI301 Machine Learning (Lecture - Hall A3)
  - 02:30 PM – 05:30 PM: AI301 ML Practical Lab (Lab 304)
- **Wednesday**:
  - 09:00 AM – 10:15 AM: AI405 LLMs & RAG (Lecture - Hall A2)
  - 10:30 AM – 11:45 AM: CS204 Algorithms (Lecture - Hall A1)
  - 05:00 PM – 06:30 PM: ACM Student Chapter Meeting (Lab 304)

---

## 2. Academic Calendar (Academic Year 2026-2027)

### 2.1 Fall Semester 2026
- **Student Orientation & Registration**: August 14 – August 17, 2026
- **Classes Commencement**: August 18, 2026
- **Fee Payment Deadline**: August 15, 2026
- **Mid-Term Examination Week**: October 12 – October 17, 2026
- **Autumn Break**: October 18 – October 25, 2026
- **End-Semester Practical Exams**: November 23 – November 28, 2026
- **End-Semester Theory Exams**: December 1 – December 15, 2026
- **Winter Vacation**: December 16, 2026 – January 11, 2027

### 2.2 Spring Semester 2027
- **Classes Commencement**: January 12, 2027
- **Fee Payment Deadline**: January 15, 2027
- **NexusHacks 36-Hour Hackathon**: February 20 – February 22, 2027
- **Mid-Term Examination Week**: March 8 – March 13, 2027
- **Aetheria Cultural Fest**: March 26 – March 28, 2027
- **End-Semester Examinations**: May 3 – May 18, 2027
- **Annual Graduation Ceremony**: June 5, 2027

---

## 3. Campus Facility Operational Hours

### 3.1 Library Hours
- **Regular Workdays (Mon–Fri)**: 08:00 AM – 11:00 PM
- **Weekends (Sat–Sun)**: 10:00 AM – 08:00 PM
- **Exam Period (Mid-terms & End-terms)**: **24 Hours / 7 Days Open** (Reading Lounge & Study Pods)

### 3.2 Student Health & Counseling Center
- **Regular Hours**: 08:30 AM – 07:00 PM (Mon–Sat)
- **24/7 Emergency Medical Hotline**: +1 (555) 019-NEXUS`
  },
  {
    id: 'faculty_directory',
    title: 'Faculty Directory',
    category: 'Faculty',
    iconName: 'GraduationCap',
    lastUpdated: '2026-09-12',
    content: `# Faculty Directory & Contact Information - Nexus Institute of Technology (NIT)

## 1. Department of Computer Science & Engineering

### 1.1 Dr. Elena Vance
- **Designation**: Professor & Head of Department (HoD), CSE
- **Academic Qualifications**: Ph.D. in Computer Science (Stanford University), M.S. in AI (MIT)
- **Specialization / Research Areas**: Large Language Models, Retrieval-Augmented Generation (RAG), Vector Embeddings, Natural Language Processing.
- **Office Location**: Building A - Room 402
- **Email**: \`evance@nexus.edu\` | Phone: +1 (555) 012-4020
- **Office Hours for Student Consultation**: Monday & Wednesday: 02:00 PM – 04:00 PM

### 1.2 Prof. Marcus Sterling
- **Designation**: Associate Professor, CSE
- **Specialization**: Data Structures & Algorithms, High-Performance Computing, Graph Analytics.
- **Office Location**: Building A - Room 410
- **Email**: \`msterling@nexus.edu\` | Phone: +1 (555) 012-4100
- **Office Hours**: Tuesday & Thursday: 10:00 AM – 12:00 PM

### 1.3 Prof. Aris Thorne
- **Designation**: Assistant Professor, CSE & ACM Student Chapter Faculty Sponsor
- **Specialization**: Artificial Intelligence Ethics, Machine Learning Engineering, Open Source Systems.
- **Office Location**: Building A - Room 415
- **Email**: \`athorne@nexus.edu\` | Phone: +1 (555) 012-4150
- **Office Hours**: Wednesday & Friday: 03:00 PM – 05:00 PM

---

## 2. Department of Cybersecurity & Data Science

### 2.1 Dr. Sophia Chen
- **Designation**: Associate Professor, Cybersecurity & Cryptography
- **Specialization**: Zero-Trust Architecture, Post-Quantum Cryptography, Web Security.
- **Office Location**: Building B - Room 205
- **Email**: \`schen@nexus.edu\` | Phone: +1 (555) 013-2050
- **Office Hours**: Tuesday: 02:00 PM – 05:00 PM

### 2.2 Dr. Vikram Patel
- **Designation**: Associate Professor, Robotics & Autonomous Systems
- **Specialization**: Computer Vision, SLAM, Drone Navigation, Autonomous Vehicles.
- **Office Location**: Building C - Room 108
- **Email**: \`vpatel@nexus.edu\` | Phone: +1 (555) 014-1080
- **Office Hours**: Monday & Thursday: 11:00 AM – 01:00 PM`
  },
  {
    id: 'rules_regulations_policies',
    title: 'Rules & Policies',
    category: 'Rules',
    iconName: 'ShieldAlert',
    lastUpdated: '2026-09-22',
    content: `# Campus Rules, Regulations & Academic Policies - Nexus Institute of Technology (NIT)

## 1. Attendance & Examination Policy

### 1.1 Minimum Attendance Requirement
- **Mandatory Threshold**: Students must maintain a minimum of **75% attendance** in each registered course (both lectures and laboratory sessions) to be eligible to appear for the end-semester examinations.
- **Medical & Special Condonation**: Attendance between **65% and 74%** may be condoned solely on valid medical grounds, supported by a official medical certificate verified by the Campus Health Center and approved by the Academic Dean.
- **Debarment**: Students with less than 65% attendance will be debarred from writing end-semester exams and must re-register for the course in a subsequent semester.

---

## 2. Code of Academic Integrity & Anti-Plagiarism

### 2.1 Plagiarism & AI Usage Regulations
- **Academic Honesty**: All submitted assignments, code repositories, laboratory reports, and research projects must be the original work of the student.
- **Generative AI Policy**: Generative AI tools (including ChatGPT, Copilot, RAG systems) may be used for learning and brainstorming only when explicitly permitted by the course instructor. Unattributed AI generation in submitted graded assignments is classified as academic dishonesty.
- **Penalties for Violation**:
  - **First Offense**: Zero mark ('0') assigned for the specific assignment or exam paper, along with a written reprimand on the student's record.
  - **Second Offense**: Formal disciplinary hearing, course failure ('F' grade), and potential 1-semester suspension.

---

## 3. Hostel & Campus Residency Code

### 3.1 Gate Timings & Curfew
- **Hostel Entry Cut-off**: Campus main gates close at **10:30 PM**.
- **Late Pass Protocol**: Students requiring late entry (up to 11:30 PM) for academic library research or campus projects must submit a digital late pass via the NIT Student Mobile App at least **4 hours prior**.
- **Overnight Absence**: Night-out permission requires written approval from parents/guardians submitted to the Chief Warden via the portal 24 hours in advance.

### 3.2 Guest Policy
- External visitors are allowed only in common lounge areas between 09:00 AM and 07:00 PM. Visitors are strictly prohibited inside hostel rooms.

---

## 4. Campus Safety & Anti-Ragging Policy

### 4.1 Anti-Ragging Mandate
- Nexus Institute of Technology maintains a **Zero-Tolerance Policy** towards ragging, harassment, or bullying in any form.
- **Reporting Mechanism**: Confidential complaints can be filed 24/7 at \`antiragging@nexus.edu\` or via the toll-free helpline \`1-800-NEXUS-SAFE\`.
- Violation results in immediate expulsion from the institute and registration of a police complaint as per government regulations.`
  }
];

import pandas as pd

# Raw chunk text
chunk_text = """Sheet: students
3336 David Palmer 19 sean43@hotmail.com Mathematics 3.16 2026
8774 Andrew Roach 23 vbecker@harvey.com Chemistry 3.75 2027
1396 Jonathan Gonzalez 22 hollydavis@gmail.com Physics 2.95 2027
6716 Kenneth Morrow 24 ganderson@wheeler-atkins.info Physics 3.55 2029
8830 Kaitlyn Martinez 18 hayesdiane@gmail.com Chemistry 2.29 2025
5305 Tiffany Wolf 23 qanderson@taylor.com Mathematics 3.3 2029
5048 James Reyes 20 drodriguez@nguyen-cooper.info Chemistry 2.44 2029
5986 Samantha Sellers 20 michelle27@hubbard-webster.com Mathematics 3.44 2025
8721 Michael Kim 25 jacksonhannah@miles.com Computer Science 3.27 2027
6622 Emily Davis 18 george02@hotmail.com Physics 3.09 2030
3254 Adam Evans 20 bryantmargaret@hernandez.com Physics 3.42 2026
8021 Alejandra Galloway 20 danielpowers@schmidt.com Chemistry 2.33 2027
9477 Cindy Johnston 22 james96@webb.org Mathematics 3.76 2025
9144 Taylor Krueger II 20 hleon@smith.net Computer Science 2.02 2025
4854 Kimberly Vang 23 carolyn34@yahoo.com Computer Science 3.35 2030
3690 Alex Allen 24 shawkaren@smith.net Physics 3.21 2029
4462 Dawn Caldwell 24 kelly96@gmail.com Computer Science 3.92 2027
3255 Connie Cline 22 patelmercedes@green.com Chemistry 3.42 2024
5446 Christine Smith 21 nicholasmejia@hotmail.com Computer Science 3.43 2027
8125 Dana Burke 23 davismichael@yahoo.com Physics 2.05 2025
3052 Suzanne Lopez 18 thomasbennett@gmail.com Computer Science 2.33 2025
2602 Meagan Thompson 23 robert13@gmail.com Physics 3.68 2028
1866 Erica Welch 25 hernandezbrent@delgado-terry.com Mathematics 3.84 2028
4908 Debra Reilly 21 amy78@jackson.com Chemistry 3.82 2027
6711 Rachel Hill 22 elizabeth41@hill-johnson.com Computer Science 2.99 2026
1165 Lauren Hill 22 timothy00@lopez.com Biology 2.49 2024
7657 Andrea Castro 19 valvarez@hotmail.com Mathematics 3.16 2024
4776 Sarah Cruz 24 melissa92@hicks.com Physics 3.56 2026
5711 Teresa Patrick PhD 25 hbrooks@nelson-bennett.info Physics 3.58 2030
8511 Brittany Thompson 19 chavezallison@yahoo.com Chemistry 2.82 2024
4009 Sean Wells 21 barreracourtney@nguyen.com Physics 2.98 2029
7998 Robert Simmons 24 daniel00@gmail.com Chemistry 3.21 2030
5676 William Holt 22 warrensmith@yahoo.com Chemistry 2.15 2030
8134 Brandon Garcia 19 brandon58@cummings-johnson.biz Computer Science 2.73 2030
5119 Alexander Jones 25 emily28@hotmail.com Physics 2.92 2024
7304 William Anderson 25 jacob29@gmail.com Mathematics 3.51 2024
3663 Shawna Glover 19 alison62@barrett.com Biology 2.75 2028
6423 Susan Galvan 22 theresa04@gmail.com Chemistry 3.26 2028
5155 Jason King 24 oliviawalls@hotmail.com Chemistry 3.74 2024
1066 Alexander Simmons 19 jrodriguez@cox.info Biology 3.26 2028
5773 James Moore 24 daniel72@long.com Physics 2.37 2024
8417 Frank Garza 19 jennifer82@cruz.com Mathematics 2.67 2027
4235 Cynthia Parker DVM 22 jasminelewis@washington.com Mathematics 2.02 2028
7002 Robert Smith 18 amanda74@gmail.com Chemistry 2.7 2028
5998 Marie Valdez 18 xwarner@nelson.com Computer Science 3.5 2030
9661 Patty Duke 18 sparksanna@gmail.com Chemistry 3.6 2027
5144 Teresa Lucas 20 epatel@edwards-smith.info Biology 3.47 2029
3719 Melanie Lowe 25 nicolepruitt@bell-gibson.org Computer Science 3.05 2030
9310 Isaiah Morales 25 lutzjason@neal.biz Physics 3.06 2025
9271 Connor Fry 19 masseytom@gmail.com Chemistry 3.94 2030
4603 Autumn Nguyen 22 john53@hotmail.com Mathematics 2.13 2030
5001 Jasmin Pena 23 ajackson@hotmail.com Chemistry 3.64 2025
5618 Mrs. Carol Williams MD 22 janice01@yahoo.com Physics 3.56 2027
3205 Andrew Lutz 20 patrick16@yahoo.com Chemistry 3.16 2029
8431 Julie Martin 25 angelawilliams@gmail.com Chemistry 3.56 2030
4604 Lisa Brown 19 annette64@gmail.com Physics 2.81 2025
7287 Holly King 23 williamsjack@randolph.com Chemistry 2.94 2024
3481 Ricky Sanders 19 buckleyandrew@gmail.com Computer Science 3.8 2027
3068 Gary Taylor 21 veronicarogers@yahoo.com Mathematics 2.62 2030
4514 Arthur Thornton 25 kirstenowen@yahoo.com Biology 2.1 2024
2499 Eric Leonard 23 theresarosales@gmail.com Biology 2.39 2027
5212 William Smith 23 daniel93@lee-morris.com Mathematics 3.29 2028
5550 Daniel Daniel 22 christianbrown@yahoo.com Physics 3.51 2027
8265 Joel Nicholson 24 michaelpratt@murray.net Chemistry 2.79 2027
6398 Benjamin Burke 18 colemary@cochran.net Computer Science 2.47 2025
2706 Brandi Roberts 24 joshua76@yahoo.com Computer Science 3.88 2025
9410 Patrick Anderson 19 meganlewis@butler-bright.com Computer Science 2.75 2029
"""

# Clean up the chunk text (removing headers, etc.)
chunk_text = chunk_text.split("\n")[1:]  # Removing the "Sheet: students" line

# Define columns for the data
columns = ['StudentID', 'Name', 'Age', 'Email', 'Department', 'GPA', 'GraduationYear']

# Parse the data into a dataframe
data = []

for line in chunk_text:
    parts = line.split()
    student_id = parts[0]
    # Split name into first and last if applicable
    name = " ".join(parts[1:-2])  # Everything between the first ID and last two columns
    age = parts[-4]
    email = parts[-3]
    department = parts[-2]
    gpa = parts[-1]
    graduation_year = parts[-1]

    # Append parsed data to the list
    data.append([student_id, name, age, email, department, gpa, graduation_year])

# Create a DataFrame
df = pd.DataFrame(data, columns=columns)

# Convert DataFrame to CSV
df.to_csv("students_data.csv", index=False)

print(df)  # This will display the readable table in your console

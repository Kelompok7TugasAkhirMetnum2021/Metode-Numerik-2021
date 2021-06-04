# Metode-Numerik-2021
KELOMPOK 7
1. Deera Herdi Mardhiyah - 26050119130067
2. Rizky Hafid Nugroho - 26050219130071
3. Raffy Bagus Prayudha - 26050119130069
4. Niken Wien Kautsar - 26050119130073
5. Dyas Isti Anggraeni - 26050119130064
6. Zhafira Mazaya P. - 26050119130066
7. Mohammad Pandu Pradana - 26050119130074
8. Oda Gracia A.S. - 26050119130070
9. Salma Nabila K - 26050119130063
10. Salmaa Bayrus - 26050119130068
11. Natasya Ayu Pramesti - 26050119130065
12. Sri Lestari - 26050119130072

MODUL 2
1. Metode Setengah Interval
Metode setengah interval atau metode bisection adalah cara menyelesaikan persamaan non-linier dengan membagi dua nilai x1 dan x2 dilakukan berulang-ulang sampai nilai x lebih kecil dari nilai tolerasi yang ditentukan. Metode bisection melakukan pengamatan terhadap nilai f(x) dengan berbagai nilai x, yang mempunyai perbedaan tanda. Taksiran akar diperhalus dengan cara membagi dua (2) pada interval x yang mempunyai beda tanda tersebut. Sedangkan metode iterasi fixed point dijalankan dengan cara membuat fungsi f(x) menjadi bentuk fungsi implisit f(x)=0 kemudian x=g(x), iterasi yang digunakan adalah dalam bentuk persamaan; xn+1 = g(xn).

2. Metode Interpolasi Linear
Interpolasi linier merupakan salah satu metode yang digunakan untuk mengetahui nilai dari suatu interval dua buah titik yang terletak dalam satu garis lurus. Metode inidilakukan dengan cara melakukan percobaan satu persatu, dari awal pendeteksian titik, yang kemudian dikembangkan dari titik-titik yang berdekatan, lalu hasil dari titik-titik yang didapatkan dicoba untuk diinterpolasikan dengan titik yang lain, dimana nantinya akan terbentuk suatu garis sesuai dengan aturan interpolasi linier yang ada atau tidak. 

3. Metode Newton-Raphson
Metode Newton-Raphson sering konvergen dengan cepat, terutama apabila iterasi dimulai cukup dekat dengan akar yang diinginkan. Namun bila iterasi dimulai jauh dari akar yang dicari, metode ini menimbulkan nilai yang jauh dari perhitungan. Pengaruh pemilihan nilai awal terhadap akar yang dicari pada metode Newton-Raphson yaitu jika nilai awal yang dipilih cukup dekat dengan nilai akar tunggal, maka dari sekian simulasi yang dilakukan hasilnya akan menampilkan akar ganda. Metode Newton-Raphson memiliki kelebihan dimana lebih cepat konvergensi dalam menentukan akar persamaan.

4. Metode Secant
Metode Secant merupakan metode yang terbentuk ketika terdapat permasalahan yang muncul pada metode Newton-Raphson. di mana pada metode Newton-Raphson terjadi kesulitan dalam mencari turunan pertama dari fungsi yang dicari. Sehingga metode secant digunakan untuk menghindari persoalan tersebut. Pada metode secant, turunan pertama adalah turunan numerik mundur.

MODUL 3
1. Metode Gauss
Metode Eliminasi Gauss adalah metode pengoperasian nilai-nilai dalam matriks menjadi matriks yang lebih sederhana dan banyak digunakan untuk penyelesaian sistem persamaan linier. Metode ini melakukan operasi baris menjadi matriks eselon-baris, yang mana mengubah persamaan linier menjadi matriks augmentasi, lalu mengoperasikannya. Metode ini termasuk metode langsung, dengan cara mengubah persamaan dalam bentuk matriks, lalu menyederhanakannya menjadi bentuk segitiga atas, kemudian mensubstitusi balik untuk mendapat nilai akar persamaannya.

2. Metode Gauss Jordan
Metode Gauss-Jordan merupakan salah satu metode matematika yang dikhususkan pada pemrograman persamaan linear. Metode ini dapat menghitung dan menentukan suatu persamaan linear yang memiliki satu sampai sembilan variabel. Gauss Jordan adalah salah satu metode untuk menyelesaikan suatu persamaan linear dengan mengubah persamaan linear menjadi matriks augmentasi, kemudian membagi elemen diagonal dan eleman sisi kanan di setiap baris dnegan elemen diagonal pada baris, buat setiap elemen diagonal sama dengan satu, sehingga didapatkan hasil pada pembagian setiap elemennya. Dalam eliminasi Gauss-Jordan, tujuannya adalah mengubah matriks koefisien menjadi matriks diagonal dan angka nol dimasukkan ke dalam matriks satu kolom. Kemudian menghilangkan elemen-elemen di atas dan di bawah elemen diagonal dari kolom yang berikutnya setiap kali melewati matriks.

3. Metode Gauss Seidel
Metode iterasi Gauss-Seidel adalah metode yang menggunakan proses iterasi hingga diperoleh nilai-nilai yang berubah-ubah dan akhirnya relatif konstan. Metode iterasi Gauss-Seidel dikembangkan dari gagasan metode iterasi pada solusi persamaan tak linier. Dengan metode iterasi Gauss-Seidel toleransi pembulatan dapat diperkecil karena iterasi dapat diteruskan sampai seteliti mungkin sesuai dengan batas toleransi yang diinginkan.

4. Metode Jacobi
Metode Jacobi merupakan salah satu metode penyelesaian sistem persamaan linear berdimensi banyak. Untuk matriks dengan dimensi kecil (kurang atau sama dengan dua), lebih efektif diselesaikan dengan aturan eliminasi. Metode Iterasi Jacobi merupakan salah satu metode tak langsung, yaitu bermula dari suatu hampiran penyelesaian awal dan kemudian berusaha memperbaiki hampiran dalam tak berhingga namun langkah konvergen. Metode Iterasi Jacobi ini digunakan untuk menyelesaikan persamaan linear berukuran besar dan proporsi koefisien nolnya besar.

MODUL 4
1. Metode Trapesium Banyak Pias
Metode trapesium merupakan metode pendekatan integral numerik dengan persamaan polinomial order satu. Dalam metode ini kurve lengkung dari fungsi f (x) digantikan oleh garis lurus. Pada metode trapesium satu pias memberikan kesalahan sangat besar. Untuk mengurangi kesalahan yang terjadi maka kurve lengkung didekati oleh sejumlah garis lurus, sehingga terbentuk banyak pias. Luas bidang adalah jumlah dari luas beberapa pias tersebut. Semakin kecil pias yang digunakan, hasil yang didapat menjadi semakin teliti. Metode trapesium dapat digunakan untuk integral suatu fungsi yang diberikan dalam bentuk numerik pada interval diskret.

2. Metode Simpson 1/3
Adalah aturan yang cukup populer dari sekian banyak metode integrasi. Kaidah simpson 1/3 adalah kaidah yang mencocokkan polinomial derajat 2 pada tiga titik data diskrit yang mempunyai jarak yang sama. Hampiran nilai integrasi yang lebih baik dapat ditingkatkan dengan menggunakan polinom interpolasi berderajat yang lebih tinggi. Aturan Simpson 1/3 bisa diadaptasi untuk N genap interval. Metode ini  lebih sederhana dan lebih akurat dibandingkan metode trapesium.

MODUL 5

1. Metode Euler
Metode Euler pada dasarnya merupakan metode yang merepresentasikan solusinya dengan beberapa suku deret Taylor. Metode ini menjadi salah satu dari metode satu langkah yang paling sederhana. Dibanding dengan beberapa metode lainnya, metode ini paling kurang teliti. Namun demikian metode ini perlu dipelajari mengingat kesederhanaannya dan mudah pemahamannya sehingga memudahkan dalam mempelajari metode lain yang lebih teliti.

2. Metode Heun
Metode Heun merupakan salah satu peningkatan dari metode Euler. Metode ini melibatkan 2 buah persamaan. Persamaan pertama disebut sebagai persamaan prediktor yang digunakan untuk memprediksi nilai integrasi awal. Persamaan kedua disebut sebagai persamaan korektor yang mengoreksi hasil integrasi awal. Metode Heun merupakan metode prediktor-korektor satu tahapan. Akurasi integrasi dapat ditingkatkan dengan melakukan koreksi ulang terhadap nilai koreksi semula menggunakan persamaan kedua.

Kami dari Kelompok 7 mengucapkan terimakasih kepada bapak ibu dosen pengampu mata kuliah metode numerik, kakak asisten, dan teman-teman oseanografi 2019. Terkhusus teman-teman kelompok 7 yang sudah bekerja sama dengan baik, sehingga laporan tugas akhir ini dapat terselesaikan dengan baik. Mohon maaf apabila banyak kekurangan dan kesalahan dari kami baik yang disengaja maupun yang tidak disengaja.

감사합니다
Thank you
Terima kasih

Sumber :
Asminah dan Sahifitri Vivi. 2012. Implementasi dan Analisis Tingkat Akurasi Software Penyelesaian Bersamaan Non Linier dengan Metode Fixed Point Iteration dan Metode Bisection. Seminar Nasional Informatika, UPN Veteran, Yogyakarta.
Sasongko, S.B. 2010.  Metode Numerik dengan Scilab. ANDI, Yogyakarta
Batarius, P., 2018. Nilai Awal Pada Metode Newton-Raphson yang Dimodifikasi dalam Penentuan Akar Persamaan. Pi: Mathematics Education Journal, 1(3), pp.108-115.
Indo, Liputan, Tony Darmanto, Kartono. 2019. Perancangan Aplikasi Perhitungan Sistem Persamaan Linear Menggunakan Metode Gauss Jordan Berbasis Android. Jurnal Masitika Vol.3 :1-12
Nurdin, A., & Hastuti, S. 2020. Analisa Gerakan Osilator Harmonik Teredam Menggunakan Metode Numerik. Journal of Mechanical Engineering. 3(2):13-19.
Silmi, dan R. Anugrahwaty. 2017. Implementasi Metode Eliminasi Gauss pada Rangkaian Listrik Menggunakan Matlab. JITEKH, 6(1): 30-35.
Oktaviani, Rizka, Bayu Prihandono, dan Helmi. 2014. Penyelesaian Numerik Sistem Persamaan Diferensial Non Linear dengan Metode Heun Pada Model Lotka-Volterra. Buletin Ilmiah Math, Volume 3 (1) : 29 – 38.
Munir, R. (2010). Metode Numerik. Bandung: Infomatika
Setyono, A. dan Sendi Novianto. 2013. Penerapan Interpolasi Linier Untuk Deteksi Garis Lurus pada Citra Gambar. Techno.COM, 12(3) : 143-149.

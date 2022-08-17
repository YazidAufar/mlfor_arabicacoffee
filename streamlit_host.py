import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from PIL import Image

# model = tf.keras.models.load_model("saved_model/mdl_wts.hdf5")
model = tf.keras.models.load_model("saved_model/MobileNetV2-Arabica Coffe-100.0.hdf5")

# Create sidebar
st.sidebar.markdown("<div><h1 style='display:inline-block'>🌱 Bantu Tani</h1></div>", unsafe_allow_html=True)
st.sidebar.markdown("Dashboard ini memungkinkan kamu mendeteksi 🔎 penyakit pada daun kopi arabika menggunakan Python and Streamlit.")
st.sidebar.markdown("👋 Kontributor : <ol><li>Yazid Aufar, M.Kom.</li> <li>Muhammad Helmy Abdillah, S.P, M.P</li> <li>Jiki Romadoni, M.Kom.</li></ol>",unsafe_allow_html=True)
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("<h5 style='text-align: center; color:black;'>Powared by</h3>", unsafe_allow_html=True)
st.sidebar.image("images/Logo Polhas.png", use_column_width=True)
st.sidebar.markdown("<h6 style='text-align: center; color: black;'>Copyright &copy; 2022</h4>", unsafe_allow_html=True)


st.markdown("<h3 style='text-align: center;'>👨‍🌾 Deteksi Penyakit Daun Kopi Arabika</h3>", unsafe_allow_html=True)
st.write("")
st.write("""
        <p>📃 Rule :</p>
        <ul>
            <li>Deteksi ini hanya digunakan untuk gambar penyakit daun kopi arabika</li>
            <li>Akurasi dalam program ini > 98%</li>
            <li>Hasil output dalam program ini ada 5, yaitu : Daun Sehat, Penambang Daun Kopi atau <i>Coffee Leaf Miner</i>, Karat Kopi atau <i>Coffee Rust</i>, Bercak Daun <i>Cercospora</i>, dan Bercak Daun <i>Phoma</i>.</li>   
        </ul>        
        """, unsafe_allow_html=True)
st.write("")

### load file
uploaded_file = st.file_uploader("🤳 Choose a image file", type="jpg")

coffee_dict = {0: 'Bercak Daun _Cercospora_',
               1: 'Daun Sehat',
               2: 'Karat Kopi atau _Coffee Rust_',
               3: 'Penambang Daun Kopi atau _Coffee Leaf Miner_',
               4: 'Bercak Daun _Phoma_'}

coffee_description = {  0:  """
                                Walaupun mungkin menyerang pohon kopi di semua usia, _Cercospora_ 
                                adalah penyakit yang dapat menyerang bibit di pembiakan dan tanaman 
                                berusia muda hingga berakibat parah. Penyakit ini biasa disebut 
                                _brown eye spot_ (noda cokelat di daun), _berry blotch_, dan _Cercospora 
                                Blotch_, disebabkan oleh jamur _Cercospora coffeicola Berk et Cook_.
                                Penyakit ini menyerang daun kopi dan buah kopi. Pada daun kopi, 
                                serangan Cercospora menimbulkan gejala mungulnya noda bulat berwarna 
                                coklat dengan pusat berwarna keabu-abuan, seringkali dikelilingi 
                                dengan lingkaran berwarna kuning. Pada buah, penyakit ini menyebabkan 
                                nekrotik, noda tertekan, berwarna cokelat kehitaman, memanjang hingga tiang ujung buah.
                                Kerusakan disebabkan oleh _cercospora leaf spot_ meliputi:
                                1. Absisi daun dan pertumbuhan kerdil pada bibit yang dibiakkan, 
                                serta defoliasi (pengguguran daun) dan retardasi pada tanaman dewasa;
                                2. Absisi buah dan keringnya cabang-cabang pada tanaman baru;
                                3. Pada tanaman produktif menyebabkan penuaan prematur dan jatuhnya 
                                buah secara prematur, menyebabkan kerusakan kualitatif dan kuantitatif 
                                pada produk akhir.                           
                            """,
                        1:  'Daun Sehat',
                        2:  """
                                Penyakit ini dikenal dengan nama _Coffee Leaf Rust_ (CLR) atau karat 
                                daun kopi, dan disebabkan oleh jamur _Hemeileia vastatrix Berk. Et Br._, 
                                dan mungkin merupakan penyakit yang paling ditakuti di budidaya kopi, 
                                karena biasa muncul di seluruh penjuru dunia, dan menyebabkan defoliasi 
                                (bergugurannya daun), yang akan mempengaruhi _yield_ kebun. Segera setelah 
                                kemunculuannya di kebun kopi komersial, karat daun kopi menghancurkan 
                                industri kopi di Ceylon (sekarang Sri Langka). Di Amerika tengah, karat 
                                kopi saat ini disebut sebagai penyakit tanaman kopi yang paling penting. 
                                Gejala dari penyakit ini adalah bercak melingkar berwarna oranye pada 
                                permukaan _inferior_ (_dorsal_) daun kopi, menampakkan _uredospora_ yang berbentuk 
                                seperti bubuk. Pada tahap lebih lanjut, beberapa bagian dari jaringan daun 
                                akan hancur dan mengalami _nekrosis_.
                            """,
                        3:  """ 
                                Penambang daun kopi atau _coffee leaf miner_ adalah hama 
                                utama tanaman kopi, khususnya di Neotropik, dimana _L. coffeella_ 
                                mendominasi, dan Afrika timur, dimana _L. caffeina_ dan _L. meyrocki_ 
                                menyerang kopi dan banyak tanaman di keluarga _Rubiaceae_. 
                                Tidak seperti _L. coffeella_ dan _L. meyrocki_, _L. caffeina_ 
                                diasosiasikan dengan kopi di sistem _shade-grown_. Pada bentuk 
                                dewasanya, penambang daun kopi berbentuk kepompong kecil yang larvanya 
                                memakan jaringan mesofil dari daun kopi, yang mana di sana juga _feses_ 
                                (kotoran) mereka disimpan. Keberadaan dari penambang daun kopi ditandai 
                                dengan area yang terkena nekrotik yang mengurangi area foliar/daun untuk 
                                fotosintesis, menyebabkan gugurnya daun yang terinfeksi, terutama di 
                                waktu-waktu tahun paling kering. Jika persentase daun yang rusak lebih 
                                dari 30%, tindakan pengendalian harus dilakukan agar keberlanjutan 
                                produktivitas dan ekonomis tidak terganggu. Namun demikian, jika pada 
                                daun-daun yang rusak 40% daun itu menunjukkan tanda adanya tawon predator, 
                                pengendalian secara kimiawi tidak perlu dilakuikan karena kontrol alami dari 
                                tawon predator tersebut sudah cukup.
                            """,
                        4:  """
                                _Phoma Leaf Spot_, juga dikenal sebagai daun terbakar atau hawar, adalah penyakit 
                                yang menyebabkan _defoliasi_ (bergugurannya daun), jatuhnya tunas buah dan kelopak 
                                bunga, dahan yang kering, dan dengan demikian menyebabkan berkurangnya produktivitas. 
                                Gejala daun adalah nampaknya noda lingkaran hitam yang bisa jadi dikelilingi 
                                lingkaran halo. Walaupun _Phoma tarda_ biasa ditemukan di Afrika, Asia Tenggara, dan 
                                Brazil, _Phoma costarricensis_ utamanya terdapat di Amerika Tengah, namun telah ditemukan 
                                juga di India, Papua Nugini, dan di altitude tinggi di Brazil. Penyakit ini lebih banyak 
                                terjadi pada tanaman yang terekspos pada angin besar yang dingin, dimana penetrasi jamur 
                                dimudahkan oleh kerusakan mekanis pada tanaman (serangga, gesekan pada daun disebabkan 
                                oleh angin atau bahkan operasi panen).
                            """}

coffee_help = {         0:  """
                                _Cercospora leaf spot_ dapat dicegah dan dikendalikan dengan pengelolaan 
                                tanaman yang bagus, baik di tempat nursery ataupun di kebun. Tindakan 
                                di nursery meliputi:
                                1. Perlindungan bibit dari angin dingin dengan pagar di pinggir;
                                2. Penggunaan substrat dengan komposisi yang direkomendasikan agar 
                                bibit dapat tumbuh kuat dan tahan terhadap penyakit ini;
                                3. Pengendalian irigasi dan pencahayaan di tempat nursery.
                                Pada fase penanaman dan produksi: 
                                1. Hindari penanaman pada tanah yang berpasir, atau tanah yang buruk, 
                                terkompresi, atau tanah yang padat;
                                2. Pertahankan suplai nutrisi tanaman yang cukup dan seimbang dengan 
                                pengendalian pupuk;
                                3. Kendalikan penyakit dengan menggunakan fungisida, terutama jika 
                                penanaman dilakukan di akhir musim hujan;
                                4. Kelola tanaman dengan baik, hindari kerusakan atau malformasi sistem 
                                akar yang dapat secara tidak langsung mempengaruhi makanan tanaman dan 
                                mendukung pertumbuhan _Cercospora_.
                                Pengendalian kimiawi harus diaplikasikan, baik kepada tanaman produktif, 
                                bibit, atau tanaman yang baru ditanam, ketika semua cara pencegahan 
                                (pemupukan, halangan angin, dll.) telah dilakukan dan terbukti tidak 
                                cukup untuk mengurangi intensitas serangan penyakit ini. 
                            """,
                        1:  'Daun Sehat',
                        2:  """
                                Untuk mengendalikan karat kopi, sangat pentung untuk dicatat bahwa semakin besar 
                                vegetasi tanaman, semakin tinggi pula infestasi residual dari penyakit ini. 
                                Juga semakin tinggi beban buah, semakin intens serangan penyakit ini. Lebih 
                                lanjut, tanaman yang ditanam dengan jarak yang ketat menimbulkan iklim mikro 
                                dengan kelembaban tinggi yang membantu penyebaran karat kopi. Dengan demikian, 
                                setelah anjuran untuk mengaplikasikan pemupukan yang seimbang, sangat juga 
                                dianjurkan untuk menggunakan kultivar yang lebih toleran terhadap penyakit, 
                                pengelolaan aliran udara yang cukup, menjarangkan pucuk untuk membantu pergerakan 
                                udara diantara tanaman, dan mengaplikasikan fungisida ketika dibutuhkan.
                            """,
                        3:  """ Tindakan pengendalian seperti pembangunan halangan angin, pengendalian gulma, 
                                dan irigasi sangat disarankan. Di daerah-daerah yang kering atau di kebun dengan 
                                sedikit teduhan, tindakan pengendalian yang disarankan untuk mengatasi serangan 
                                _L. meyrocki_ dan  _L. coffeella_ adalah dengan menggunakan jarak tanam yang lebih 
                                kecil antar tanamannya. Hal ini meningkatkan kelembaban udara di sekitar tanaman, 
                                sehingga mengurangi kerumunan atau infestasi serangga.Karena kerusakan oleh 
                                penambang daun kopi ini sudah sangat menyebar, banyak usaha telah dilakukan untuk 
                                mengembangkan kultivar yang resisten terhadap penambang daun kopi. Salah satu 
                                contoh dari usaha ini adalah kultivar Siriema 842, dan kerja kolaboratif telah 
                                menyempurnakan proses pembiakan vegetatifnya untuk distribusi skala besar.
                            """,
                        4:  """
                                Tindakan pengendalian yang disarankan meliputi menghindari area yang sering 
                                diterjang angin dingin ketika membangun kebun, berhati-hati dalam membuat halangan 
                                angin baik itu yang permanen atau temporer, dan aplikasi pupuk dan fungisida yang 
                                seimbang (sebelum dan setelah pembungaan) di waktu-waktu yang mendukung serangan 
                                penyakit. 
                            """}

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    # 128 untuk model yang di jupyter notebook Image Detection -  Arabica Coffee
    resized = cv2.resize(opencv_image,(128,128))
    # 224 untuk model yang di google colab
    # resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        # st.title('Sepertinya gambar ini adalah {}'.format(indo_dict[prediction]))
        st.markdown('Kami cukup yakin bahwa ini adalah:')
        st.header(format(coffee_dict[prediction]))
        st.caption(format(coffee_description[prediction]))
        with st.expander("🍂 Tindakan pengendalian "):
                st.caption(format(coffee_help[prediction]))

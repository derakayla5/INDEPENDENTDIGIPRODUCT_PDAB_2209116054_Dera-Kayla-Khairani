import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.naive_bayes import GaussianNB

# Fungsi untuk menampilkan dataframe
def show_dataframe(data):
    st.write(data)

# Fungsi untuk menampilkan visualisasi korelasi matriks
def show_correlation_heatmap(df, columns):
    if len(columns) < 2:
        st.warning("Pilih setidaknya dua kolom untuk menampilkan korelasi.")
        return
    
    st.header("Heatmap Korelasi Matriks")
    st.write("Visualisasi ini menampilkan korelasi antar kolom dalam bentuk matriks.")
    
    # Menghitung matriks korelasi
    correlation_matrix = df[columns].corr(method='pearson')
    
    # Menampilkan heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax)
    st.pyplot(fig)


# Memuat DataFrame dari file CSV
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Memanggil fungsi untuk memuat data
df = load_data('energycleaned.csv')

# Menampilkan judul Streamlit
st.title("Prediksi Kebutuhan Energi Pendingin berdasarkan Desain Bangunan")

# Sidebar
st.sidebar.title('Menu')
menu = st.sidebar.radio('', ['Beranda', 'Visualisasi', 'Prediksi'])


#BERANDA
# Konten berdasarkan menu yang dipilih
if menu == 'Beranda':
    # Tampilkan gambar
    st.image('https://i.pinimg.com/564x/2a/d2/67/2ad26730c764173a492b891559cf8149.jpg')

    # Tampilkan penjelasan atau ringkasan bisnis
    
    st.title("Business Understanding")
    st.write("Business Objective")
    st.write("Tujuan bisnis dari dataset ini adalah untuk memberikan pemahaman yang lebih baik tentang hubungan antara desain bangunan dan kebutuhan energi pendinginan, dengan tujuan meningkatkan efisiensi, kualitas, dan memberikan desain bangunan yang lebih ramah lingkungan.")
    st.write("Assess Situation")
    st.write("Situasi bisnis yang mendasari analisis ini adalah semakin kesini, semakin banyak orang meminta properti yang ramah lingkungan dan efisien secara energi di tengah kesadaran global akan perlunya keberlanjutan.")
    st.write("Data Mining Goals")
    st.write("Tujuan dari data mining pada dataset ini adalah membantu pemilik bangunan agar dapat mengoptimalkan desain bangunan mereka untuk mencapai efisiensi energi yang lebih tinggi dan merancang bangunan yang lebih nyaman serta ramah lingkungan. Ini dapat meningkatkan kualitas hidup penghuninya dan mengurangi dampak lingkungan negatif yang dihasilkan oleh konsumsi energi yang tinggi.")
    st.write("Project Plan")
    st.write("Project Plan Proyek ini akan mencakup langkah-langkah mulai dari pengumpulan data, pemahaman data, hingga analisis regresi untuk melihat bagaimana desain bangunan memengaruhi kebutuhan energi pendinginan. Hasilnya akan dievaluasi dan disajikan dengan cara yang mudah dimengerti, serta akan dipertimbangkan untuk menerapkan rekomendasi desain yang dihasilkan.")
    
    
    

    
    
    
    
#VISUALISASI    
#VISUALISASI    
elif menu == 'Visualisasi':
    st.title('Visualisasi Data')
    
    # Pilih kolom untuk menampilkan korelasi
    selected_columns = st.multiselect("Pilih kolom untuk menampilkan korelasi:", df.columns)
    # Tampilkan heatmap korelasi matriks
    show_correlation_heatmap(df, selected_columns)

    # Isi konten untuk Visualisasi
    st.write('Menampilkan Visualisasi Data') 

    # Split layout menjadi dua kolom
    col1, col2 = st.columns(2)

    # Plot scatter plot Surface Area vs Cooling Load di kolom pertama
    with col1:
        fig_scatter = plt.figure(figsize=(8, 6))
        plt.scatter(df['Surface_Area'], df['Cooling_Load'], alpha=0.5)
        plt.title('Scatter Plot: Surface Area vs Cooling Load')
        plt.xlabel('Surface Area')
        plt.ylabel('Cooling Load')
        plt.grid(True)
        st.pyplot(fig_scatter)
        st.write("Dapat dilihat dari Variabel Luas Bangunan dan Cooling Area menunjukkan adanya korelasi positif antara surface dengan cooling area. jadi, semakin besar luas permukaan atau surface area tadi, maka semakin tinggi beban pendinginnya.")

    # Plot histogram Wall Area di kolom kedua
    with col2:
        fig_hist = plt.figure(figsize=(8, 6))
        plt.hist(df['Wall_Area'], bins=20, color='skyblue', edgecolor='black')
        plt.title('Distribusi Wall Area')
        plt.xlabel('Wall Area')
        plt.ylabel('Frekuensi')
        st.pyplot(fig_hist)
        st.write("Dapat dilihat pada histogram tersebut Rentang nilai pada kolom wall area yang paling sering muncul adalah 300 dan 325")

    # Plot Bar Chart Horizontal Roof Area & Cooling Load
    # Buat bar chart horizontal di kolom pertama
    with col1:
        fig_bar = plt.figure(figsize=(8, 6))
        rata_rata_Roof_Area = df['Roof_Area'].mean()
        rata_rata_Cooling_Load = df['Cooling_Load'].mean()
        plt.title('Bar Chart Roof Area & Cooling Load')
        plt.barh(['Roof_Area', 'Cooling_Load'], [rata_rata_Roof_Area, rata_rata_Cooling_Load], color=['skyblue', 'orange'])
        plt.xlabel('Rata-rata')
        plt.ylabel('Kolom')
        st.pyplot(fig_bar)
        st.write("Dapat dilihat bahwa rata rata Kolom Roof Area lebih tinggi daripada kolom Cooling load yaitu 175")

    # Plot Pie Chart Heating Load Category di kolom kedua
    with col2:
        fig_pie_heating = plt.figure(figsize=(8, 6))
        heating_load_counts = df['Heating_Load_Category'].value_counts()
        plt.pie(heating_load_counts, labels=heating_load_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Pie Chart for Heating Load Category')
        plt.axis('equal')  # Memastikan lingkaran berbentuk bulat
        st.pyplot(fig_pie_heating)
        st.write("Dapat dilihat bahwa dalam Kategori Heating Load Energy lebih banyak Presentase untuk Heating Load yang Rendah atau diwakilkan dengan 0. sedangkan Heating load yang Tinggi diwakilkan oleh 1")

    # Plot Pie Chart Cooling Load Category di kolom kedua
    with col2:
        fig_pie_cooling = plt.figure(figsize=(8, 6))
        Cooling_load_counts = df['Cooling_Load_Category'].value_counts()
        plt.pie(Cooling_load_counts, labels=Cooling_load_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Pie Chart for Cooling Load Category')
        plt.axis('equal')  # Memastikan lingkaran berbentuk bulat
        st.pyplot(fig_pie_cooling)
        st.write("Dapat dilihat bahwa dalam Kategori Cooling Load Energy lebih banyak Presentase untuk Cooling Load yang tinggi atau diwakilkan dengan 1. sedangkan Cooling load yang Rendah diwakilkan oleh 0")


    
    
#PREDIKSI
elif menu == 'Prediksi':
    st.title('Prediksi dengan Model GNB')
# Isi konten untuk Prediksi
#Load Model From File
    try:
        file_path = 'gnb.pkl'
        with open(file_path, 'rb') as f:
            clf = joblib.load(f) 
    except FileNotFoundError:
        st.error("File model tidak ditemukan. Pastikan file model ada dan nama filenya benar.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")

    # Fungsi untuk membuat input data
    def input_data():
        st.header("Masukkan Data untuk Prediksi")
        inputs = {}
        for col in ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area', 'Overall_Height', 'Glazing_Area','Heating_Load','Cooling_Load']:
            inputs[col] = st.number_input(f'Masukkan {col}', min_value=0.0, step=0.01)
        return pd.DataFrame([inputs])

    # Prediksi dengan model GNB
    def predict_with_gnb(data):
        prediction = clf.predict(data)
        return prediction

    # Input data
    input_df = input_data()

    # Tombol untuk melakukan prediksi
    if st.button('Prediksi'):
        try:
            # Prediksi dengan model GNB
            result = predict_with_gnb(input_df)
            st.write('Hasil Prediksi:')
            st.write(result)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {str(e)}")
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
    
# # Tampilkan heatmap
#     st.header('Heatmap')
#     st.write('Heatmap digunakan untuk menunjukkan korelasi antara variabel dalam dataset.')
#     sns.heatmap(df.corr(method='pearson'), annot=True, cmap='coolwarm')
#     st.pyplot()





# elif menu == 'Visualisasi':
#     st.title('Visualisasi Data')
#     # Isi konten untuk Visualisasi
#     st.write('Tambahkan konten visualisasi data Anda di sini.')






# elif menu == 'Prediksi':
#     st.title('Prediksi')
#     # Isi konten untuk Prediksi
#     st.write('Tambahkan konten prediksi Anda di sini.')





































# import streamlit as st

# # Sidebar
# st.sidebar.title('Menu')
# menu = st.sidebar.radio('', ['Beranda', 'Visualisasi', 'Prediksi'])

# # Konten berdasarkan menu yang dipilih
# if menu == 'Beranda':
#     st.title('Beranda (EDA)')
#     # Isi konten untuk Beranda (EDA)
#     st.write('Tambahkan konten EDA Anda di sini.')

# elif menu == 'Visualisasi':
#     st.title('Visualisasi Data')
#     # Isi konten untuk Visualisasi
#     st.write('Tambahkan konten visualisasi data Anda di sini.')

# elif menu == 'Prediksi':
#     st.title('Prediksi')
#     # Isi konten untuk Prediksi
#     st.write('Tambahkan konten prediksi Anda di sini.')

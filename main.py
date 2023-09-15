from src.data_loading import DataLoader


def main():
    print("Hola mundo")

    INSTACE_ID = '98e90c95-04e9-4aa8-a105-688da74595be'
    SH_CLIENT_ID = '1c7168a0-37b5-444f-a7e6-826ae6c19d90'
    SH_CLIENT_SECRET = 'f&}UR;bV(I)fx?r|:hlNZ0sK1utD4ny_4V0WsQzJ'

    loader = DataLoader(INSTACE_ID,SH_CLIENT_ID,SH_CLIENT_SECRET)
    loader.get_image(-4.447581,-54.980870)
    loader.display_random_samples(3)




if __name__ == "__main__":
    main()

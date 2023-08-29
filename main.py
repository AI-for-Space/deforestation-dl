from src.data_loading import DataLoader


def main():
    print("Hola mundo")

    INSTACE_ID = '3e8b80d9-d9eb-43e1-84c7-44a569e6ba83'
    SH_CLIENT_ID = '570d0ec5-4d9b-4852-8d30-45e1af205e89'
    SH_CLIENT_SECRET = 'rXF2Lz4|-%yHoe2dPBgp{e-10-[8s?X3*m:)&2r}'

    loader = DataLoader(INSTACE_ID,SH_CLIENT_ID,SH_CLIENT_SECRET)
    loader.get_image()




if __name__ == "__main__":
    main()

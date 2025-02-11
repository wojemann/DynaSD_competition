if __name__ == '__main__':
    import argparse
    from DynaSD_scalp.main import main
    parser = argparse.ArgumentParser()
    
    parser.add_argument("input")
    parser.add_argument("output")

    args = parser.parse_args()
    main(args.input,args.output)
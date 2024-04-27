import subprocess

def execute_file(file_path):
    try:
        subprocess.run(["python", file_path], check=True)
    except subprocess.CalledProcessError as e:
        print("파일 실행 중 오류가 발생했습니다:", e)

def main():
    while True:
        user_input = input("키를 입력하세요 (f 또는 d): ")
        command = user_input.split()[0]  # 입력된 문자열에서 명령어만 추출
        if command == 'f':   # fast mode 
            file_path = r"C:/Users/USER/Downloads/Visual_assistant/hf_cap_/application/app_mpyy(main).py"
            execute_file(file_path)
        elif command == 'd': # detail mode 
            file_path = r"C:/Users/USER/Downloads/Visual_assistant/visual test_/open_renew/open_flamingo/application/app_mpyy(main).py"
            execute_file(file_path)
        else:
            print("올바른 키를 입력하세요.")

if __name__ == "__main__":
    main()
from calculator.genshin import user

if __name__ == '__main__':
    is_exit = 0
    curr_user = None
    while is_exit != 1:
        if curr_user is None:
            user_account = str(input("Please enter your account: "))
            user_passcode = str(input("Please enter your password: "))
            curr_user = user.init_genshin_user(user_account, user_passcode)

        next_step = str(input("Please enter next step: "))
        if next_step == "9":
            is_exit = 1

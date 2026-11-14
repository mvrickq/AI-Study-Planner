import asyncio
import os
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()
BASE_URL = os.getenv("BLACKBOARD_BASE_URL")


async def main():
    if not BASE_URL:
        raise ValueError("BLACKBOARD_BASE_URL not set in .env")

    async with async_playwright() as p:
        # Visible browser so you can log in and approve MFA
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        print(f"Opening Blackboard login page: {BASE_URL}")
        await page.goto(BASE_URL)

        print(
            "\n👉 Log in normally (including Microsoft Authenticator).\n"
            "   Once you see your Blackboard dashboard or activity stream,\n"
            "   come back to this terminal and press ENTER.\n"
        )
        input("Press ENTER here once you are fully logged in... ")

        # Save cookies + tokens to auth_state.json
        await context.storage_state(path="auth_state.json")
        print("✅ Saved authenticated session to auth_state.json")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())

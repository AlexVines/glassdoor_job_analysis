import glassdoor_scraper as gs

path = r"C:\Users\Alex\PycharmProjects\glassdoor_job_analysis\chromedriver.exe"

df = gs.get_jobs('data scientist', 1000, False, path, 15)

df.to_csv('glassdoor_jobs.csv', index=False)